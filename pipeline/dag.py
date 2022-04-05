import os
import sys
import logging
import threading
import multiprocessing
import queue as Queue
from typing import Union

from proto import pipeline_service_pb2
from op.operator import RequestOp, ResponseOp, VirtualOp
from pipeline.channel import ProcessChannel, ThreadChannel
from pipeline.error_catch import ErrorCatch, ParamChecker
from pipeline.channel import ChannelData, ChannelDataErrCode, ChannelStopError, ChannelDataType
from pipeline.util import ThreadIdGenerator, PipelineProcSyncManager, NameGenerator
from pipeline.profiler import TimeProfiler, PerformanceTracer
from utils.logger import get_logger, set_logger_level

_LOGGER = get_logger("Serving")


class DAGExecutor(object):
    """
    DAG (有向无环图) 执行器，DAG 服务入口
    """

    def __init__(self, response_op, server_conf, worker_idx):
        """
        初始化 DAGExecutor

        Args:
            response_op: 响应算子。Response Op
            server_conf: 服务配置。config.yml
            worker_idx: DAGExecutor index，当 _build_dag_each_worker 为 True时
                        PipelineServer 会创建许多 DAGExecutor
        """
        build_dag_each_worker = server_conf["build_dag_each_worker"]
        server_worker_num = server_conf["worker_num"]
        dag_conf = server_conf["dag"]

        self._retry = dag_conf["retry"]
        self._server_use_profile = dag_conf["use_profile"]
        self._enable_prometheus = False  # 普罗米修斯，性能数据的监控
        if "enable_prometheus" in dag_conf:
            self._enable_prometheus = dag_conf["enable_prometheus"]
        if "prometheus_port" in dag_conf and self._enable_prometheus:
            self._prometheus_port = dag_conf["prometheus_port"]
        else:
            self._prometheus_port = None
        channel_size = dag_conf["channel_size"]
        channel_recv_first_arrive = dag_conf["channel_recv_first_arrive"]
        self._is_thread_op = dag_conf["is_thread_op"]

        tracer_conf = dag_conf["tracer"]
        tracer_interval_s = tracer_conf["interval_s"]

        self.name = "@DAGExecutor"
        self._profiler = TimeProfiler()
        self._profiler.enable(True)

        self._tracer = None
        if tracer_interval_s >= 1:
            self._tracer = PerformanceTracer(self._is_thread_op, tracer_interval_s, server_worker_num)
            if self._enable_prometheus:
                self._tracer.set_enable_dict(True)

        self._dag = DAG(self.name, response_op, self._server_use_profile,
                        self._prometheus_port, self._is_thread_op, channel_size,
                        build_dag_each_worker, self._tracer, channel_recv_first_arrive)
        in_channel, out_channel, pack_rpc_func, unpack_rpc_func = self._dag.build()
        self._dag.start()

        self._set_in_channel(in_channel)
        self._set_out_channel(out_channel)
        self._pack_rpc_func = pack_rpc_func
        self._unpack_rpc_func = unpack_rpc_func

        if self._tracer is not None:
            self._tracer.start()

        # 生成 id
        # data_id: 服务唯一 ID，框架自动生成
        # log_id: 跟踪业务请求，可以为空，不唯一
        base_counter = 0
        gen_id_step = 1
        self._id_generator = ThreadIdGenerator(
            max_id=1000000000000000000,
            base_counter=base_counter,
            step=gen_id_step)

        self._cv_pool = {}  # {data_id: cond_v}
        self._cv_for_cv_pool = threading.Condition()
        self._fetch_buffer = {}
        self._receive_func = None

        self._client_profile_key = "pipeline.profile"
        self._client_profile_value = "1"

    @ErrorCatch
    def start(self):
        """
        启动一个线程在后台从最后的 channel 接收数据

        Returns:
            None
        """

        self._receive_func = threading.Thread(target=DAGExecutor._receive_out_channel_func, args=(self,))
        self._receive_func.daemon = True
        self._receive_func.start()
        _LOGGER.debug("[DAG Executor] start receive thread")

    def stop(self):
        """
        停止 DAG

        Returns:
            None
        """
        self._dag.stop()
        self._dag.join()
        _LOGGER.info("[DAG Executor] Stop")

    def _get_next_data_id(self):
        """
        Generate data_id incrementally and Uniquely

        Args:
            None

        Returns:
            data_id: uniq id
            cond_v: condition variable
        """
        data_id = self._id_generator.next()
        cond_v = threading.Condition()
        with self._cv_for_cv_pool:
            self._cv_pool[data_id] = cond_v
            self._fetch_buffer[data_id] = None
        return data_id, cond_v

    def _set_in_channel(self, in_channel: Union[ThreadChannel, ProcessChannel]):
        """
        Set in_channel of DAG

        Args:
            in_channel: input channel of DAG

        Returns:

        """
        if not isinstance(in_channel, (ThreadChannel, ProcessChannel)):
            _LOGGER.critical("[DAG Executor] Failed to set in_channel: "
                             "in_channel must be Channel type, but get {}".
                             format(type(in_channel)))
            os._exit(-1)

        self._in_channel = in_channel
        _LOGGER.info("[DAG] set in channel succ, name [{}]".format(self.name))

    def _set_out_channel(self, out_channel: Union[ThreadChannel, ProcessChannel]):
        """
        Set out_channel of DAG

        Args:
            out_channel: output channel of DAG

        Returns:

        """
        if not isinstance(out_channel, (ThreadChannel, ProcessChannel)):
            _LOGGER.critical("[DAG Executor] Failed to set out_channel: "
                             "must be Channel type, but get {}".format(
                type(out_channel)))
            os._exit(-1)
        out_channel.add_consumer(self.name)
        self._out_channel = out_channel

    def _receive_out_channel_func(self):
        """
        从输出 channel 接收数据，并将数据放入 _fetch_buffer。
        函数 _get_channeldata_from_fetch_buffer 按重试次数获取数据

        Returns:

        """
        cv = None
        while True:
            try:
                channeldata_dict = self._out_channel.front(self.name)
            except ChannelStopError:
                _LOGGER.info("[DAG Executor] Stop.")
                with self._cv_for_cv_pool:
                    for data_id, cv in self._cv_pool.items():
                        closed_errror_data = ChannelData(
                            error_code=ChannelDataErrCode.CLOSED_ERROR.value,
                            error_info="dag closed.",
                            data_id=data_id)
                        with cv:
                            self._fetch_buffer[data_id] = closed_errror_data
                            cv.notify_all()
                break
            if len(channeldata_dict) != 1:
                _LOGGER.critical("[DAG Executor] Failed to fetch result: out_channel cannot have multiple input ops")
                os._exit(-1)
            (_, channeldata), = channeldata_dict.items()
            if not isinstance(channeldata, ChannelData):
                _LOGGER.critical(
                    "[DAG Executor] Failed to fetch result: data in out_channel "
                    "must be ChannelData type, but get {}".format(type(channeldata)))
                os._exit(-1)

            data_id = channeldata.id
            _LOGGER.debug("(logid={}) [recive thread] Fetched data".format(
                data_id))
            with self._cv_for_cv_pool:
                cond_v = self._cv_pool[data_id]
            with cond_v:
                self._fetch_buffer[data_id] = channeldata
                cond_v.notify_all()

    def _get_channeldata_from_fetch_buffer(self, data_id, cond_v):
        """
        从 _fetch_buffer 获取 channeldata

        Args:
            data_id: search key
            cond_v: conditional variable

        Returns:

        """

        ready_data = None
        with cond_v:
            with self._cv_for_cv_pool:
                if self._fetch_buffer[data_id] is not None:
                    # 请求的数据已经准备好
                    ready_data = self._fetch_buffer[data_id]
                    self._cv_pool.pop(data_id)
                    self._fetch_buffer.pop(data_id)
            if ready_data is None:
                # 等待数据准备好
                cond_v.wait()
                with self._cv_for_cv_pool:
                    ready_data = self._fetch_buffer[data_id]
                    self._cv_pool.pop(data_id)
                    self._fetch_buffer.pop(data_id)

        _LOGGER.debug("(data_id={}) [resp thread] Got data".format(data_id))
        return ready_data

    def _pack_channeldata(self, rpc_request, data_id):
        """
        从 RPC 请求 unpacking 数据，并创建一条 channeldata

        Args:
           rpc_request: 一条 RPC 请求
           data_id: data id, unique

        Returns:
            ChannelData: one channel data to be processed
        """
        dictdata = None
        log_id = None
        try:
            dictdata, log_id, prod_error_code, prod_error_info = self._unpack_rpc_func(rpc_request)
        except Exception as e:
            _LOGGER.error("(logid={}) Failed to parse RPC request package: {}".format(data_id, e), exc_info=True)
            return ChannelData(
                error_code=ChannelDataErrCode.RPC_PACKAGE_ERROR.value,
                error_info="rpc package error: {}".format(e),
                data_id=data_id,
                log_id=log_id)
        else:
            # 因为 unpack_rpc_func 由用户重写，需要查看返回的 product_error_code
            # 和 rpc_request 中的 client_profile_key 字段
            if prod_error_code is not None:
                # 业务错误发生
                _LOGGER.error("unpack_rpc_func prod_error_code:{}".format(prod_error_code))
                return ChannelData(
                    error_code=ChannelDataErrCode.PRODUCT_ERROR.value,
                    error_info="",
                    prod_error_code=prod_error_code,
                    prod_error_info=prod_error_info,
                    data_id=data_id,
                    log_id=log_id)

            profile_value = None
            profile_value = dictdata.get(self._client_profile_key)
            client_need_profile = (profile_value == self._client_profile_value)
            return ChannelData(
                datatype=ChannelDataType.DICT.value,
                dictdata=dictdata,
                data_id=data_id,
                log_id=log_id,
                client_need_profile=client_need_profile)

    def call(self, rpc_request):
        if self._tracer is not None:
            trace_buffer = self._tracer.data_buffer()

        data_id, cond_v = self._get_next_data_id()
        start_call, end_call = None, None
        if not self._is_thread_op:
            start_call = self._profiler.record("call_{}#DAG-{}_0".format(data_id, data_id))
        else:
            start_call = self._profiler.record("call_{}#DAG_0".format(data_id))

        self._profiler.record("prepack_{}#{}_0".format(data_id, self.name))
        req_channeldata = self._pack_channeldata(rpc_request, data_id)
        self._profiler.record("prepack_{}#{}_1".format(data_id, self.name))

        log_id = req_channeldata.log_id
        _LOGGER.info("(data_id={} log_id={}) Succ Generate ID ".format(data_id, log_id))

        resp_channeldata = None
        for i in range(self._retry):
            _LOGGER.debug("(data_id={}) Pushing data into Graph engine".format(data_id))
            try:
                if req_channeldata is None:
                    _LOGGER.critical("(data_id={} log_id={}) req_channeldata is None".format(data_id, log_id))
                if not isinstance(self._in_channel,
                                  (ThreadChannel, ProcessChannel)):
                    _LOGGER.critical(
                        "(data_id={} log_id={})[DAG Executor] Failed to "
                        "set in_channel: in_channel must be Channel type, but get {}".format(
                            data_id, log_id, type(self._in_channel)))
                self._in_channel.push(req_channeldata, self.name)
            except ChannelStopError:
                _LOGGER.error("(data_id:{} log_id={})[DAG Executor] Stop".format(data_id, log_id))
                with self._cv_for_cv_pool:
                    self._cv_pool.pop(data_id)
                return self._pack_for_rpc_resp(
                    ChannelData(
                        error_code=ChannelDataErrCode.CLOSED_ERROR.value,
                        error_info="dag closed.",
                        data_id=data_id))

            _LOGGER.debug("(data_id={} log_id={}) Wait for Graph engine...".format(data_id, log_id))
            resp_channeldata = self._get_channeldata_from_fetch_buffer(data_id, cond_v)

            if resp_channeldata.error_code == ChannelDataErrCode.OK.value:
                _LOGGER.info("(data_id={} log_id={}) Succ predict".format(data_id, log_id))
                break
            else:
                _LOGGER.error("(data_id={} log_id={}) Failed to predict: {}".format(
                    data_id, log_id, resp_channeldata.error_info))
                if resp_channeldata.error_code != ChannelDataErrCode.TIMEOUT.value:
                    break

            if i + 1 < self._retry:
                _LOGGER.warning("(data_id={} log_id={}) DAGExecutor retry({}/{})".format(
                    data_id, log_id, i + 1, self._retry))

        _LOGGER.debug("(data_id={} log_id={}) Packing RPC response package".format(data_id, log_id))
        self._profiler.record("postpack_{}#{}_0".format(data_id, self.name))
        rpc_resp = self._pack_for_rpc_resp(resp_channeldata)
        self._profiler.record("postpack_{}#{}_1".format(data_id, self.name))
        if not self._is_thread_op:
            end_call = self._profiler.record("call_{}#DAG-{}_1".format(data_id,
                                                                       data_id))
        else:
            end_call = self._profiler.record("call_{}#DAG_1".format(data_id))

        if self._tracer is not None:
            trace_buffer.put({
                "name": "DAG",
                "id": data_id,
                "succ": resp_channeldata.error_code == ChannelDataErrCode.OK.value,
                "actions": {"call_{}".format(data_id): end_call - start_call, },
            })

        profile_str = self._profiler.gen_profile_str()
        if self._server_use_profile:
            sys.stderr.write(profile_str)

        # add profile info into rpc_resp
        if resp_channeldata.client_need_profile:
            profile_set = resp_channeldata.profile_data_set
            profile_set.add(profile_str)
            profile_value = "".join(list(profile_set))
            rpc_resp.key.append(self._client_profile_key)
            rpc_resp.value.append(profile_value)

        return rpc_resp

    def _pack_for_rpc_resp(self, channeldata):
        """
        打包一条 RCP 响应

        Args:
            channeldata: 要被打包的 channeldata

        Returns:
            resp: 一条 RPC 响应
        """
        try:
            return self._pack_rpc_func(channeldata)
        except Exception as e:
            _LOGGER.error(f"(logid={channeldata.id}) Failed to pack RPC response package: {e}", exc_info=True)
            resp = pipeline_service_pb2.Response()
            resp.error_no = ChannelDataErrCode.RPC_PACKAGE_ERROR.value
            resp.error_msg = "rpc package error: {}".format(e)
            return resp


class DAG(object):
    """
    Directed Acyclic Graph(DAG 有向无环图) 引擎，构建一个 DAG 拓扑。
    """

    def __init__(self, request_name, response_op, use_profile, prometheus_port, is_thread_op,
                 channel_size, build_dag_each_worker, tracer, channel_recv_first_arrive):
        _LOGGER.debug("[DAG] init one DAG")
        _LOGGER.info("{}, {}, {}, {}, {}, {} ,{} ,{} ,{}".format(
            request_name, response_op, use_profile, prometheus_port, is_thread_op, channel_size,
            build_dag_each_worker, tracer, channel_recv_first_arrive))

        @ErrorCatch
        @ParamChecker
        def init_helper(self, request_name, response_op, use_profile,
                        prometheus_port, is_thread_op, channel_size,
                        build_dag_each_worker, tracer, channel_recv_first_arrive):
            _LOGGER.debug("[DAG] init_helper()")
            self._request_name = request_name
            self._response_op = response_op
            self._use_profile = use_profile
            self._prometheus_port = prometheus_port
            self._use_prometheus = (self._prometheus_port is not None)
            self._is_thread_op = is_thread_op
            self._channel_size = channel_size
            self._build_dag_each_worker = build_dag_each_worker
            self._tracer = tracer
            self._channel_recv_first_arrive = channel_recv_first_arrive
            if not self._is_thread_op:
                self._manager = PipelineProcSyncManager()

        init_helper(self, request_name, response_op, use_profile, prometheus_port, is_thread_op,
                    channel_size, build_dag_each_worker, tracer, channel_recv_first_arrive)

        print("[DAG] Success init")
        _LOGGER.info("[DAG] Success init")

    @staticmethod
    def get_use_ops(response_op: ResponseOp):
        """
        从 ResponseOp 开始递归地遍历前面的 ops。获取每个 op (除了 ResponseOp) 使用的所有 ops 和 post op 列表

        Args:
            response_op: ResponseOp

        Returns:
            used_ops: used ops, set
            succ_ops_of_use_op: op and the next op list, dict.

        """

        unique_names = set()
        used_ops = set()
        succ_ops_of_use_op = {}  # {op_name: succ_ops}
        que = Queue.Queue()
        que.put(response_op)
        while que.qsize() != 0:
            op = que.get()
            for pred_op in op.get_input_ops():
                if pred_op.name not in succ_ops_of_use_op:
                    succ_ops_of_use_op[pred_op.name] = []
                if op != response_op:
                    succ_ops_of_use_op[pred_op.name].append(op)
                if pred_op not in used_ops:
                    que.put(pred_op)
                    used_ops.add(pred_op)
                    # 检测 op 的名称是否全局唯一
                    if pred_op.name in unique_names:
                        _LOGGER.critical("Failed to get used Ops: the name of Op must be unique: {}".format(
                            pred_op.name))
                        os._exit(-1)
                    unique_names.add(pred_op.name)
        return used_ops, succ_ops_of_use_op

    def _gen_channel(self, name_gen):
        """
        Generate one ThreadChannel or ProcessChannel.

        Args:
            name_gen: channel name

        Returns:
            channel: one channel generated
        """
        channel = None
        if self._is_thread_op:
            channel = ThreadChannel(
                name=name_gen.next(),
                maxsize=self._channel_size,
                channel_recv_first_arrive=self._channel_recv_first_arrive)
        else:
            channel = ProcessChannel(
                self._manager,
                name=name_gen.next(),
                maxsize=self._channel_size,
                channel_recv_first_arrive=self._channel_recv_first_arrive)
        _LOGGER.debug("[DAG] Generate channel: {}".format(channel.name))
        return channel

    def _gen_virtual_op(self, name_gen):
        """
        生成一个虚拟的 Op

        Args:
            name_gen: Op name

        Returns:
            vir_op: one virtual Op object.
        """
        vir_op = VirtualOp(name=name_gen.next())
        _LOGGER.debug("[DAG] Generate virtual_op: {}".format(vir_op.name))
        return vir_op

    def _topo_sort(self, used_ops, response_op: ResponseOp, out_degree_ops):
        """
        DAG 拓扑排序，创建倒立的多层视图

        Args:
            used_ops: DAG 中使用的 Ops
            response_op: 响应 Op
            out_degree_ops: 每个 Op 的后继 Op 列表, dict 类型. get_use_ops() 的输出

        Returns:
            dag_views: 反向的分层拓扑列表. 例如:
                DAG :[A -> B -> C -> E]
                            \-> D /
                dag_views: [[E], [C, D], [B], [A]]

            last_op: ResponseOp 前的最后一个 Op
        """
        out_degree_num = {name: len(ops) for name, ops in out_degree_ops.items()}
        que_idx = 0  # 滚动队列 scroll queue
        ques = [Queue.Queue() for _ in range(2)]
        zero_indegree_num = 0
        for op in used_ops:
            if len(op.get_input_ops()) == 0:
                zero_indegree_num += 1
        if zero_indegree_num != 1:
            _LOGGER.critical("Failed to topo sort: DAG contains multiple RequestOps")
            os._exit(-1)
        last_op = response_op.get_input_ops()[0]
        ques[que_idx].put(last_op)

        # 拓扑排序得到 dag 视图
        dag_views = []
        sorted_op_num = 0
        while True:
            que = ques[que_idx]
            next_que = ques[(que_idx + 1) % 2]
            dag_view = []
            while que.qsize() != 0:
                op = que.get()
                dag_view.append(op)
                sorted_op_num += 1
                for pred_op in op.get_input_ops():
                    out_degree_num[pred_op.name] -= 1
                    if out_degree_num[pred_op.name] == 0:
                        next_que.put(pred_op)
            dag_views.append(dag_view)
            if next_que.qsize() == 0:
                break
            que_idx = (que_idx + 1) % 2
        if sorted_op_num < len(used_ops):
            _LOGGER.critical("Failed to topo sort: not legal DAG")
            os._exit(-1)

        return dag_views, last_op

    def _build_dag(self, response_op):
        """
        构建 DAG，DAG 类中最重要的方法。核心步骤：
        1. get_use_ops: 获取当前 op 已使用的 ops 和 out degree op 列表 (出度op列表？当前 op 的 post op)
        2. _topo_sort: 拓扑排序创建倒立的多层视图
        3. 创建 channel 和虚拟 op

        Args:
            response_op: ResponseOp

        Returns:
            actual_ops: DAG 中使用的 ops，包括虚拟 ops
            channels: DAG 中使用的 channels
            input_channel: 输入 channel
            output_channel: 输出 channel
            pack_func:
            unpack_func:
        """
        if response_op is None:
            _LOGGER.critical("Failed to build DAG: ResponseOp has not been set.")
            os._exit(-1)
        used_ops, out_degree_ops = DAG.get_use_ops(response_op)
        if not self._build_dag_each_worker:
            _LOGGER.info("================= USED OP =================")
            for op in used_ops:
                if not isinstance(op, RequestOp):
                    _LOGGER.info(op.name)
            _LOGGER.info("-------------------------------------------")
        if len(used_ops) <= 1:
            _LOGGER.critical(
                "Failed to build DAG: besides RequestOp and Response Op, "
                "there should be at least one Op in DAG."
            )
            os._exit(-1)
        if self._build_dag_each_worker:
            _LOGGER.info("Because `build_dag_each_worker` mode is used, "
                         "Auto-batching is set to the default config: "
                         "batch_size=1, auto_batching_timeout=None")
            for op in used_ops:
                op.use_default_auto_batching_config()

        dag_views, last_op = self._topo_sort(used_ops, response_op, out_degree_ops)
        dag_views = list(reversed(dag_views))
        if not self._build_dag_each_worker:
            _LOGGER.info("================== DAG ====================")
            for idx, view in enumerate(dag_views):
                _LOGGER.info("(VIEW {})".format(idx))
                for op in view:
                    _LOGGER.info("  [{}]".format(op.name))
                    for out_op in out_degree_ops[op.name]:
                        _LOGGER.info("    - {}".format(out_op.name))
            _LOGGER.info("-------------------------------------------")

        # 创建 channels 和虚拟 Ops
        virtual_op_name_gen = NameGenerator("vir")
        channel_name_gen = NameGenerator("chl")
        virtual_ops = []
        channels = []
        input_channel = None
        actual_view = None
        for v_idx, view in enumerate(dag_views):
            if v_idx + 1 >= len(dag_views):
                break
            next_view = dag_views[v_idx + 1]
            if actual_view is None:
                actual_view = view
            actual_next_view = []
            pred_op_of_next_view_op = {}
            for op in actual_view:
                # 查找下一视图中成功的 Op ，并创建虚拟 Op
                for succ_op in out_degree_ops[op.name]:
                    if succ_op in next_view:
                        if succ_op not in actual_next_view:
                            actual_next_view.append(succ_op)
                        if succ_op.name not in pred_op_of_next_view_op:
                            pred_op_of_next_view_op[succ_op.name] = []
                        pred_op_of_next_view_op[succ_op.name].append(op)
                    else:
                        # 创建虚拟 Op
                        virtual_op = self._gen_virtual_op(virtual_op_name_gen)
                        virtual_ops.append(virtual_op)
                        out_degree_ops[virtual_op.name] = [succ_op]
                        actual_next_view.append(virtual_op)
                        pred_op_of_next_view_op[virtual_op.name] = [op]
                        virtual_op.add_virtual_pred_op(op)
            actual_view = actual_next_view
            # create channel
            processed_op = set()
            for o_idx, op in enumerate(actual_next_view):
                if op.name in processed_op:
                    continue
                channel = self._gen_channel(channel_name_gen)
                channels.append(channel)
                op.add_input_channel(channel)
                _LOGGER.info("op:{} add input channel.".format(op.name))
                pred_ops = pred_op_of_next_view_op[op.name]
                if v_idx == 0:
                    input_channel = channel
                else:
                    # if pred_op is virtual op, it will use ancestors as producers to channel
                    for pred_op in pred_ops:
                        pred_op.add_output_channel(channel)
                        _LOGGER.info("pred_op:{} add output channel".format(
                            pred_op.name))
                processed_op.add(op.name)
                # find same input op to combine channel
                for other_op in actual_next_view[o_idx + 1:]:
                    if other_op.name in processed_op:
                        continue
                    other_pred_ops = pred_op_of_next_view_op[other_op.name]
                    if len(other_pred_ops) != len(pred_ops):
                        continue
                    same_flag = True
                    for pred_op in pred_ops:
                        if pred_op not in other_pred_ops:
                            same_flag = False
                            break
                    if same_flag:
                        other_op.add_input_channel(channel)
                        processed_op.add(other_op.name)
        output_channel = self._gen_channel(channel_name_gen)
        channels.append(output_channel)
        last_op.add_output_channel(output_channel)
        _LOGGER.info("last op:{} add output channel".format(last_op.name))

        pack_func, unpack_func = None, None
        pack_func = response_op.pack_response_package

        actual_ops = virtual_ops
        for op in used_ops:
            if len(op.get_input_ops()) == 0:
                # set special features of the request op.
                # 1.set unpack function.
                # 2.set output channel.
                unpack_func = op.unpack_request_package
                op.add_output_channel(input_channel)
                continue
            actual_ops.append(op)

        for c in channels:
            _LOGGER.debug("\nChannel({}):\n\t- producers: {}\n\t- consumers: {}".format(
                c.name, c.get_producers(), c.get_consumers()))

        return actual_ops, channels, input_channel, output_channel, pack_func, unpack_func

    def get_channels(self):
        return self._channels

    def build(self):
        """
        构建 DAG 接口，返回 channel 和 func

        Returns:
            input_channel: 输入 channel
            output_channel: 输出 channel
            pack_func:
            unpack_func:
        """
        actual_ops, channels, input_channel, output_channel, pack_func, unpack_func = self._build_dag(
            self._response_op)
        _LOGGER.info("[DAG] Succ build DAG")

        self._actual_ops = actual_ops
        self._channels = channels
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._pack_func = pack_func
        self._unpack_func = unpack_func

        if self._tracer is not None:
            self._tracer.set_channels(self._channels)

        return self._input_channel, self._output_channel, self._pack_func, self._unpack_func

    def start_porm(self, prometheus_port):
        _LOGGER.warning("start_porm(prometheus_port) has not supported right now.")
        from flask import Response, Flask
        app = Flask(__name__)

        @app.route("/metrics")
        def requests_count():
            _LOGGER.warning("requests_count() has not supported right now.")
            pass  # TODO
            # return Response()

        def porm_run():
            app.run(host="0.0.0.0", port=prometheus_port)

        # p = threading.Thread(target=porm_run, args=())
        # _LOGGER.info("Prometheus Start")
        # p.daemon = True
        # p.start()

    def start(self):
        """
        每个 Op 根据 _is_thread_op 来启动一个线程/进程

        Args:
            None

        Returns:
            _threads_or_proces: 线程或进程列表.
        """
        _LOGGER.debug("[DAG] start()")
        self._threads_or_proces = []
        for op in self._actual_ops:
            op.use_profiler(self._use_profile)
            op.set_tracer(self._tracer)
            op.set_use_prometheus(self._use_prometheus)
            if self._is_thread_op:
                self._threads_or_proces.extend(op.start_with_thread())
            else:
                self._threads_or_proces.extend(op.start_with_process())
        _LOGGER.info("[DAG] start")
        if self._use_prometheus:
            _LOGGER.info("Prometheus Start 1")
            self.start_prom(self._prometheus_port)

        # not join yet
        return self._threads_or_proces

    def join(self):
        """
        All threads or processes join.

        Args:
            None

        Returns:
            None
        """
        for x in self._threads_or_proces:
            if x is not None:
                x.join()

    def stop(self):
        """
        停止并清理所有的 channels

        Args:
            None

        Returns:
            None
        """
        for chl in self._channels:
            chl.stop()
        for op in self._actual_ops:
            op.clean_input_channel()
            op.clean_output_channels()
