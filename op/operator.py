import os
import sys
import time
import logging
import threading
import collections
import multiprocessing
import queue as Queue
import func_timeout
from concurrent import futures
from typing import Union
from time import time as _time

import numpy as np
from client.client import Client
from proto import pipeline_service_pb2
from pipeline import local_service_handler
from pipeline.error_catch import ErrorCatch, CustomException, CustomExceptionCode, ParamChecker, ParamVerify
from pipeline.error_catch import CustomExceptionCode as ChannelDataErrCode
from pipeline.channel import ThreadChannel, ProcessChannel
from pipeline.channel import ChannelData, ChannelDataType, ChannelTimeoutError, ChannelStopError
from pipeline.profiler import TimeProfiler
from pipeline.util import NameGenerator
from utils.process_util import kill_stop_process_by_pid
from utils.logger import get_logger
from utils.time_util import timestamp as _timestamp

check_feed_dict = ParamVerify.check_feed_dict
check_fetch_list = ParamVerify.check_fetch_list

_LOGGER = get_logger()
_op_name_gen = NameGenerator("Op")


class Op(object):
    def __init__(self, name=None,
                 input_ops=[],
                 server_endpoints=None,
                 fetch_list=None,
                 client_config=None,
                 client_type=None,
                 concurrency=None,
                 timeout=None,
                 retry=0,
                 batch_size=None,
                 auto_batching_timeout=None,
                 local_service_handler=None,
                 jump_to_ops=[]):

        if name is None:
            name = _op_name_gen.next()  # 没有传入 name 则生成一个
        self.name = name  # 识别 Op 类型的，必须全局唯一
        _LOGGER.debug(f"[Op] init one Op name={self.name}")
        self.concurrency = concurrency  # 并发数
        self.set_input_ops(input_ops)
        self.set_jump_to_ops(jump_to_ops)

        self._local_service_handler = local_service_handler
        self._server_endpoints = server_endpoints
        self._fetch_names = fetch_list
        self._client_config = client_config  # 客户端配置
        self.client_type = client_type  # one of [grpc, local_predictor]
        self._timeout = timeout  # 超时时间
        self._retry = max(1, retry)  # 重试次数
        self._batch_size = batch_size
        self._auto_batching_timeout = auto_batching_timeout

        self._input = None
        self._outputs = []

        self._server_use_profile = False
        self._tracer = None

        # for grpc_pipeline predict mode. False, string key/val; True, tensor format.
        self._pack_tensor_format = False

        # only for thread op
        self._for_init_op_lock = threading.Lock()
        self._for_close_op_lock = threading.Lock()
        self._succ_init_op = False
        self._succ_close_op = False
        self.dynamic_shape_info = {}
        self.set_dynamic_shape_info()

    def set_dynamic_shape_info(self):
        """
        when opening tensorrt(configure in config.yml) and each time the input shape
        for inferring is different, using this method for configuring tensorrt
        dynamic shape to infer in each op model
        """
        pass

    # for feed/fetch dict check
    @staticmethod
    def get_feed_fetch_list(client):
        from pipeline.local_predictor import LocalPredictor
        if isinstance(client, Client):
            feed_names = client.get_feed_names()
            fetch_names = client.get_fetch_names()
        if isinstance(client, LocalPredictor):
            feed_names = client.feed_names_
            fetch_names = client.fetch_names_
        return feed_names, fetch_names

    def init_from_dict(self, conf):
        """
        从配置文件 (yaml格式) 初始化 Op。如果 server_endpoints 存在，则是远端 RPC 模式，否则是本地 RPC 模式。
        在本地 RPC 模式中有两种预测类型：grpc 和 local_predictor。

        Args:
            conf: config.yaml

        Returns:
        """
        _LOGGER.debug(f"[Op] {self.name} init_from_dict() start")
        if self.concurrency is None:
            self.concurrency = conf["concurrency"]
        if self._retry is None:
            self._retry = conf["retry"]
        if self._fetch_names is None:
            self._fetch_names = conf.get("fetch_list")
        if self._client_config is None:
            self._client_config = conf.get("client_config")
        if self._timeout is None:
            self._timeout = conf["timeout"]
        if self._timeout > 0:
            self._timeout = self._timeout / 1000.0
        else:
            self._timeout = -1

        if self._batch_size is None:
            self._batch_size = conf["batch_size"]
        if self._auto_batching_timeout is None:
            self._auto_batching_timeout = conf["auto_batching_timeout"]
        if self._auto_batching_timeout <= 0 or self._batch_size == 1:
            _LOGGER.debug(self._log(
                "Because auto_batching_timeout <= 0 or batch_size == 1, set auto_batching_timeout to None."))
            self._auto_batching_timeout = None
        else:
            self._auto_batching_timeout = self._auto_batching_timeout / 1000.0

        self.model_config = None
        self.workdir = None
        self.thread_num = self.concurrency
        self.device_type = -1
        self.devices = ""
        self.mem_optim = False
        self.ir_optim = False
        self.precision = "fp32"
        self.use_mkldnn = False
        self.mkldnn_cache_capacity = 0
        self.mkldnn_op_list = None
        self.mkldnn_bf16_op_list = None
        self.min_subgraph_size = 3
        self.use_calib = False

        if self._server_endpoints is None:
            server_endpoints = conf.get("server_endpoints", [])
            if len(server_endpoints) != 0:
                # remote service
                self.with_serving = True
                self._server_endpoints = server_endpoints
                self.client_type = conf["client_type"]
            else:
                if self._local_service_handler is None:
                    local_service_conf = conf.get("local_service_conf")
                    _LOGGER.info("[Op] init_from_dict() local_service_conf: {}".format(local_service_conf))
                    self.model_config = local_service_conf.get("model_config")
                    self.client_type = local_service_conf.get("client_type")
                    self.workdir = local_service_conf.get("workdir")
                    self.thread_num = local_service_conf.get("thread_num")
                    self.device_type = local_service_conf.get("device_type")
                    self.devices = local_service_conf.get("devices")
                    self.mem_optim = local_service_conf.get("mem_optim")
                    self.ir_optim = local_service_conf.get("ir_optim")
                    self._fetch_names = local_service_conf.get("fetch_list")
                    self.precision = local_service_conf.get("precision")
                    self.use_calib = local_service_conf.get("use_calib")
                    self.use_mkldnn = local_service_conf.get("use_mkldnn")
                    self.mkldnn_cache_capacity = local_service_conf.get("mkldnn_cache_capacity")
                    self.mkldnn_op_list = local_service_conf.get("mkldnn_op_list")
                    self.mkldnn_bf16_op_list = local_service_conf.get("mkldnn_bf16_op_list")
                    self.min_subgraph_size = local_service_conf.get("min_subgraph_size")

                    if self.model_config is None:
                        self.with_serving = False
                    else:
                        # 本地 rcp 服务
                        self.with_serving = True
                        if self.client_type == "grpc":
                            _LOGGER.debug(f"[Op] {self.name} create service_handler = LocalServiceHandler()")
                            service_handler = local_service_handler.LocalServiceHandler(
                                model_config=self.model_config,
                                client_type=self.client_type,
                                workdir=self.workdir,
                                thread_num=self.thread_num,
                                device_type=self.device_type,
                                devices=self.devices,
                                mem_optim=self.mem_optim,
                                ir_optim=self.ir_optim,
                                precision=self.precision,
                                use_mkldnn=self.use_mkldnn,
                                mkldnn_cache_capacity=self.mkldnn_cache_capacity,
                                mkldnn_op_list=self.mkldnn_bf16_op_list,
                                mkldnn_bf16_op_list=self.mkldnn_bf16_op_list,
                                min_subgraph_size=self.min_subgraph_size,
                                dynamic_shape_info=self.dynamic_shape_info,
                                use_calib=self.use_calib
                            )
                            service_handler.prepare_server()  # get fetch_list
                            service_ports = service_handler.get_port_list()
                            self._server_endpoints = [
                                "127.0.0.1:{}".format(p) for p in service_ports
                            ]
                            if self._client_config is None:
                                self._client_config = service_handler.get_client_config()
                            if self._fetch_names is None:
                                self._fetch_names = service_handler.get_fetch_list()
                        elif self.client_type == "local_predictor":
                            _LOGGER.debug(f"[Op] {self.name} create service_handler = LocalServiceHandler()")
                            service_handler = local_service_handler.LocalServiceHandler(
                                model_config=self.model_config,
                                client_type=self.client_type,
                                workdir=self.workdir,
                                thread_num=self.thread_num,
                                device_type=self.device_type,
                                devices=self.devices,
                                fetch_names=self._fetch_names,
                                mem_optim=self.mem_optim,
                                ir_optim=self.ir_optim,
                                precision=self.precision,
                                use_mkldnn=self.use_mkldnn,
                                mkldnn_cache_capacity=self.mkldnn_cache_capacity,
                                mkldnn_op_list=self.mkldnn_op_list,
                                mkldnn_bf16_op_list=self.mkldnn_bf16_op_list,
                                min_subgraph_size=self.min_subgraph_size,
                                dynamic_shape_info=self.dynamic_shape_info,
                                use_calib=self.use_calib)
                            if self._client_config is None:
                                self._client_config = service_handler.get_client_config()
                        _LOGGER.debug(f"[Op] {self.name} _local_service_handler = service_handler")
                        self._local_service_handler = service_handler
                else:
                    self.with_serving = True
                    self._local_service_handler.prepare_server()  # get fetch_list
                    service_ports = self._local_service_handler.get_port_list()
                    self._server_endpoints = ["127.0.0.1:{}".format(p) for p in service_ports]
                    if self._client_config is None:
                        self._client_config = self._local_service_handler.get_client_config()
                    if self._fetch_names is None:
                        self._fetch_names = self._local_service_handler.get_fetch_list()
        else:
            self.with_serving = True

        if not isinstance(self, RequestOp) and not isinstance(self, ResponseOp):
            _LOGGER.info(self._log(
                "\n\tinput_ops: {},"
                "\n\tserver_endpoints: {}"
                "\n\tfetch_list: {}"
                "\n\tclient_config: {}"
                "\n\tconcurrency: {},"
                "\n\ttimeout(s): {},"
                "\n\tretry: {},"
                "\n\tbatch_size: {},"
                "\n\tauto_batching_timeout(s): {}".format(
                    ", ".join([op.name for op in self._input_ops]),
                    self._server_endpoints, self._fetch_names, self._client_config,
                    self.concurrency, self._timeout, self._retry, self._batch_size,
                    self._auto_batching_timeout)))

        _LOGGER.debug(f"[Op] {self.name} init_from_dict() end")

    def launch_local_rpc_service(self):
        _LOGGER.debug(f"[Op] {self.name} launch_local_rpc_service() start")
        if self._local_service_handler is None:
            _LOGGER.warning(self._log("Failed to launch local rpc service: local_service_handler is None."))
            return
        port = self._local_service_handler.get_port_list()
        # if self._local_service_handler.client_type == "local_predictor":
        #    _LOGGER.info("Op({}) use local predictor.")
        #    return
        self._local_service_handler.start_server()
        _LOGGER.info(f"Op({self.name}) use local rpc service at port: {port}")
        _LOGGER.debug(f"[Op] {self.name} launch_local_rpc_service() end")

    def use_default_auto_batching_config(self):
        if self._batch_size != 1:
            _LOGGER.warning(f"Op({self.name}) reset batch_size=1 (original: {self._batch_size})")
            self._batch_size = 1
        if self._auto_batching_timeout is not None:
            _LOGGER.warning(
                f"Op({self.name}) reset auto_batching_timeout=None (original: {self._auto_batching_timeout})")
            self._auto_batching_timeout = None

    def use_profiler(self, use_profile):
        self._server_use_profile = use_profile

    def set_tracer(self, tracer):
        self._tracer = tracer

    def set_use_prometheus(self, use_prometheus):
        self._use_prometheus = use_prometheus

    def init_client(self, client_config, server_endpoints):
        """
        初始化客户端对象，有三种类型的客户端，brpc、grpc 和 local_predictor。
        在 grpc 和 brpc 模式中，客户端连接到端点

        Args:
            client_config: 客户端配置信息
            server_endpoints: 服务 IP/Port 列表

        Returns:
            client: 客户端对象
        """
        _LOGGER.debug("[Op] init_client() client_type={}".format(self.client_type))
        if not self.with_serving:
            _LOGGER.info("Op({}) has no client (and it also do not run the process function)".format(self.name))
            return None
        if self.client_type == "brpc":
            pass  # TODO Client()
        elif self.client_type == "pipeline_grpc":
            client = None  # TODO PPClient()
        elif self.client_type == "local_predictor":
            if self.local_predictor is None:
                raise ValueError("local predictor not yet created")
            client = self.local_predictor
            self.right_feed_names, self.right_fetch_names = self.get_feed_fetch_list(client)
        else:
            raise ValueError("Failed to init client: unknown client type {}".format(self.client_type))
        if self._fetch_names is None:
            self._fetch_names = client.fetch_names_
            _LOGGER.info(f"Op({self.name}) has no fetch name set. So fetch all vars")
        if self.client_type != "local_predictor":
            client.connect(server_endpoints)
        _LOGGER.info(f"init_client, feed_list:{self.right_feed_names}, fetch_list: {self.right_fetch_names}")
        return client

    def get_input_ops(self):
        return self._input_ops

    def set_input_ops(self, ops):
        """
        设置输入 Op，每个 Op 可以有很多输入 Ops，但只能有 1 个输入 channel.

        Args:
            ops: op 列表

        Returns:
            None.
        """
        if not isinstance(ops, list):
            ops = [] if ops is None else [ops]
        self._input_ops = []
        for op in ops:
            if not isinstance(op, Op):
                _LOGGER.critical(self._log(
                    "Failed to set input_ops: input op must be Op type, not {}".format(type(op))))
                os._exit(-1)
            self._input_ops.append(op)

    def set_jump_to_ops(self, ops):
        """
        设置跳过 Ops，然后，此 Op 将 channel 数据发送给输出 channel.

        Args:
            ops: op list to be jumpped

        Returns:
            None.
        """
        if not isinstance(ops, list):
            ops = [] if ops is None else [ops]

        self._jump_to_ops = []
        for op in ops:
            if not isinstance(op, Op):
                _LOGGER.critical(self._log(
                    "Failed to set input_ops: input op must be Op type, not {}".format(type(op))))
                os._exit(-1)
            self._jump_to_ops.append(op)

    def is_jump_op(self):
        """ Op 是否有 _jump_to_ops 成员 """
        return len(self._jump_to_ops) > 0

    def check_jumping(self, input_data):
        """
        Check whether to send data to jump ops.WhileOp needs to rewrite
        this interface. this function returns False default.

        Args:
            input_data: input data to be preprocessed

        Returns:
            True, send data to the output channel of jump ops
            False, send data to output channel.
        """
        return False

    def get_output_channels_of_jump_ops(self):
        """
        Get output channels of jump ops

        Args:
            None

        Returns:
            list of channels
        """
        channels = []
        if self.is_jump_op() is False:
            return channels
        for op in self._jump_to_ops:
            _LOGGER.info("op:{} extend op._get_output_channels:{}".format(op.name, op._get_output_channels()))
            channels.extend(op._get_output_channels())

        _LOGGER.info("get_output_channels_of_jump_ops, channels:{}".format(channels))
        return channels

    def add_input_channel(self, channel: Union[ThreadChannel, ProcessChannel]):
        """ 添加输入 channel 到 Op。每个 Op 可以有多个前驱 Op，但只有一个输入 channel """
        if not isinstance(channel, (ThreadChannel, ProcessChannel)):
            _LOGGER.critical(self._log(
                "Failed to set input_channel: input channel must be Channel type, not {}".format(type(channel))))
            os._exit(-1)
        channel.add_consumer(self.name)
        self._input = channel

    def clean_input_channel(self):
        self._input = None

    def _get_input_channel(self):
        return self._input

    def add_output_channel(self, channel: Union[ThreadChannel, ProcessChannel]):
        """ 添加输出 channel 到 Op。每个 Op 可以有多个后继 Op，但只有一个输出 channel """
        if not isinstance(channel, (ThreadChannel, ProcessChannel)):
            _LOGGER.critical(self._log(
                "Failed to add output_channel: output channel must be Channel type, not {}".format(type(channel))))
            os._exit(-1)
        channel.add_producer(self.name)
        self._outputs.append(channel)
        _LOGGER.debug("op:{} add output_channel {} {}".format(self.name, channel.name, channel))

    def clean_output_channels(self):
        self._outputs = []

    def _get_output_channels(self):
        return self._outputs

    def preprocess(self, input_dicts, data_id=0, log_id=0):
        """
        预处理，为 process 装配数据。用户可以重载本函数

        Args:
            input_dicts: 要被预处理的数据
            data_id: 内部唯一 id，自增
            log_id: 全局唯一 id for RTT，默认 0

        Return:
            output_data: 给 process 的数据
            is_skip_process: 是否跳过 process，默认 False
            prod_errcode: 默认 None，否则发生业务错误。处理方式和异常一样
            prod_errinfo: 默认 ""
        """
        # multiple previous Op
        if len(input_dicts) != 1:
            _LOGGER.critical(self._log(
                "Failed to run preprocess: this Op has multiple previous inputs. Please override this func."))
            os._exit(-1)

        (_, input_dict), = input_dicts.items()
        return input_dict, False, None, ""

    def process(self, feed_batch, typical_logid=0):
        """
        process 步骤，发送 request 到推理服务或本地预测。
        用户不需要重载本函数

        Args:
            feed_batch: 要 feed 给推理服务的数据
            typical_logid: 标志批量预测，通常是 batch 中的第一个logid，默认 0

        Returns:
            call_result: 预测结果
            error_code: 错误代码
            error_info: 错误信息
        """

        call_result = None
        error_code = ChannelDataErrCode.OK.value
        error_info = ""

        @ErrorCatch
        @ParamChecker
        def feed_fetch_list_check_helper(feed_batch, fetch_list, right_feed_names, right_fetch_names):
            # feed_batch: lambda feed_batch: check_feed_dict(feed_batch[0], self.right_feed_names),
            # fetch_list: lambda fetch_list: check_fetch_list(fetch_list, self.right_fetch_names),
            # log_id):
            check_feed_dict(feed_batch[0], right_feed_names)
            check_fetch_list(fetch_list, right_fetch_names)
            return None

        # _, resp = feed_fetch_list_check_helper(feed_batch, self._fetch_names, log_id=typical_logid)
        _, resp = feed_fetch_list_check_helper(
            feed_batch, self._fetch_names, self.right_feed_names, self.right_fetch_names)
        if resp.error_no != CustomExceptionCode.OK.value:
            error_code = resp.error_no
            error_info = resp.error_msg
            call_result = None
            return call_result, error_code, error_info

        if self.client_type == "local_predictor":
            error_code, error_info = ChannelData.check_batch_npdata(feed_batch)
            if error_code != ChannelDataErrCode.OK.value:
                _LOGGER.error(self._log(
                    "Failed to run process: {}. feed_batch must be npdata in process for local_predictor mode.".format(
                        error_info)))
                return call_result, ChannelDataErrCode.TYPE_ERROR.value, "feed_batch must be npdata"
            call_result = self.client.predict(
                feed=feed_batch[0],
                fetch=self._fetch_names,
                batch=True,
                log_id=typical_logid)
        elif self.client_type == "brpc":
            pass  # TODO
        elif self.client_type == "pipeline_grpc":
            error_code, error_info = ChannelData.check_dictdata(feed_batch)
            if error_code != 0:
                _LOGGER.error(self._log(
                    "Failed to run process: {}. feed_batch must be npdata in process for pipeline_grpc mode.".format(
                        error_info)))
                return call_result, ChannelDataErrCode.TYPE_ERROR.value, "feed_batch must be dict"

            call_result = self.client.predict(
                feed_dict=feed_batch[0],
                fetch=self._fetch_names,
                asyn=False,
                pack_tensor_format=self._pack_tensor_format,
                profile=False)
            if call_result is None:
                _LOGGER.error(self._log("Failed in pipeline_grpc. call_result is None."))
                return call_result, ChannelDataErrCode.UNKNOWN.value, "pipeline_grpc error"
            if call_result.error_no != 0:
                _LOGGER.error(self._log("Failed in pipeline_grpc. error_no:{}, error_info:{}".format(
                    call_result.error_no, call_result.error_msg)))
                return call_result, ChannelDataErrCode(call_result.error_no).value, call_result.error_msg

            new_dict = {}
            error_code = ChannelDataErrCode(call_result.error_no).value
            error_info = call_result.error_msg
            for idx, key in enumerate(call_result.key):
                new_dict[key] = [call_result.value[idx]]
            call_result = new_dict

        return call_result, error_code, error_info

    def postprocess(self, input_data, fetch_data, data_id=0, log_id=0):
        """
        postprocess 步骤，汇聚数据给下一 Op 或输出

        Args:
            input_data: preprocess 步骤返回的数据，dict (单预测) 或 list (批量预测)
            fetch_data: process 步骤返回的数据，dict (单预测) 或 list (批量预测)
            data_id: 内部唯一 id，自增
            log_id: log_id，默认 0

        Returns:
            fetch_dict: dict 类型结果
            prod_errcode: 默认 None, 否则, 业务错误发生. 它与异常处理方式一样
            prod_errinfo: 默认 ""
        """
        fetch_dict = {}
        if isinstance(fetch_data, dict):
            fetch_dict = fetch_data
        return fetch_dict, None, ""

    def _parse_channeldata(self, channeldata_dict):
        """
        解析一条 channeldata

        Args:
            channeldata_dict : 要解析的 channeldata，dict 类型

        Return:
            data_id: 由 dag._id_generator 创建，唯一
            error_channeldata: 错误的 channeldata
            parsed_data: 从 channeldata 中获取 np/dict 数据
            client_need_profile: need profile info
            profile_set: profile info
            log_id: log_id，用于追踪请求
        """

        data_id, error_channeldata = None, None
        client_need_profile, profile_set = False, set()
        parsed_data = {}

        key = list(channeldata_dict.keys())[0]
        data_id = channeldata_dict[key].id
        log_id = channeldata_dict[key].log_id
        client_need_profile = channeldata_dict[key].client_need_profile

        for name, data in channeldata_dict.items():
            if data.error_code != ChannelDataErrCode.OK.value:
                error_channeldata = data
                break
            parsed_data[name] = data.parse()
            if client_need_profile:
                profile_set |= data.profile_data_set
        return data_id, error_channeldata, parsed_data, client_need_profile, profile_set, log_id

    def _push_to_output_channels(self, data,
                                 channels,
                                 name=None,
                                 profile_str=None,
                                 client_need_profile=False,
                                 profile_set=None):
        """
        将数据放到输出 channels，不执行下一步骤 (preprocess, process, postprocess)

        Args:
            data: 要投放的 channeldata
            channels: 输出 channels
            name: op name
            profile_str: one profile message
            client_need_profile: 默认 False
            profile_set: profile message collections

        Returns:
            None
        """
        if name is None:
            name = self.name

        # add profile into channeldata
        if client_need_profile and profile_set is not None:
            if profile_str is not None:
                profile_set.add(profile_str)
            data.add_profile(profile_set)

        for channel in channels:
            channel.push(data, name)

    def start_with_process(self):
        """
        每个 Op 创建一个进程来执行 main 循环，在各自独立的进程中初始化 CUDA 环境

        Args:
            None

        Returns:
            进程列表
        """

        _LOGGER.debug("[Op] start_with_process()")
        trace_buffer = None
        if self._tracer is not None:
            trace_buffer = self._tracer.data_buffer()
        process = []
        for concurrency_idx in range(self.concurrency):
            p = multiprocessing.Process(
                target=self._run,
                args=(concurrency_idx, self._get_input_channel(),
                      self._get_output_channels(), False, trace_buffer,
                      self.model_config, self.workdir, self.thread_num,
                      self.device_type, self.devices, self.mem_optim,
                      self.ir_optim, self.precision, self.use_mkldnn,
                      self.mkldnn_cache_capacity, self.mkldnn_op_list,
                      self.mkldnn_bf16_op_list, self.is_jump_op(),
                      self.get_output_channels_of_jump_ops(),
                      self.min_subgraph_size, self.dynamic_shape_info,
                      self.use_calib))
            p.daemon = True
            p.start()
            process.append(p)
        return process

    def start_with_thread(self):
        """
        每个 Op 创建一个线程来执行 main 循环，在主线程中初始化 CUDA 环境

        Args:
            None

        Returns:
            线程列表
        """
        _LOGGER.debug("[Op] {} start_with_thread()".format(self.name))
        trace_buffer = None
        if self._tracer is not None:
            trace_buffer = self._tracer.data_buffer()

        # 在主线程中初始化 CUDA 环境
        if self.client_type == "local_predictor":
            _LOGGER.info("Init cuda env in main thread")
            self.local_predictor = self._local_service_handler.get_client(0)

        threads = []
        for concurrency_idx in range(self.concurrency):
            t = threading.Thread(
                target=self._run,
                args=(concurrency_idx, self._get_input_channel(),
                      self._get_output_channels(), True, trace_buffer,
                      self.model_config, self.workdir, self.thread_num,
                      self.device_type, self.devices, self.mem_optim,
                      self.ir_optim, self.precision, self.use_mkldnn,
                      self.mkldnn_cache_capacity, self.mkldnn_op_list,
                      self.mkldnn_bf16_op_list, self.is_jump_op(),
                      self.get_output_channels_of_jump_ops(),
                      self.min_subgraph_size, self.dynamic_shape_info,
                      self.use_calib))

            # 当进程存在时，它会尝试终止其所有守护子进程
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def init_op(self):
        pass

    def _run_preprocess(self, parsed_data_dict, op_info_prefix, logid_dict):
        """
        执行 preprocess 步骤

        Args:
            parsed_data_dict: data to be pre-processed
            op_info_prefix: input op info
            logid_dict: logid dict

        Returns:
            preped_data_dict: data preprocessed, to be processed
            err_channeldata_dict: when exceptions occurred, putting errors in it.
            skip_process_dict: skip process stage or not

        """

        _LOGGER.debug("{} Running preprocess".format(op_info_prefix))
        preped_data_dict = collections.OrderedDict()
        err_channeldata_dict = collections.OrderedDict()
        skip_process_dict = {}

        @ErrorCatch
        def preprocess_help(self, parsed_data, data_id, logid_dict):
            preped_data, is_skip_process, prod_errcode, prod_errinfo = self.preprocess(
                parsed_data, data_id, logid_dict.get(data_id))
            return preped_data, is_skip_process, prod_errcode, prod_errinfo

        for data_id, parsed_data in parsed_data_dict.items():
            preped_data, error_channeldata = None, None
            is_skip_process = False
            prod_error_code, prod_error_info = None, None
            log_id = logid_dict.get(data_id)
            process_res, resp = preprocess_help(self, parsed_data, data_id=data_id, logid_dict=logid_dict)
            if resp.error_no == CustomExceptionCode.OK.value:
                preped_data, is_skip_process, prod_error_code, prod_error_info = process_res
                if is_skip_process is True:
                    skip_process_dict[data_id] = True
                if prod_error_code is not None:
                    _LOGGER.error(
                        "data_id: {} return product error. Product ErrNo:{}, Product ErrMsg: {}".format(
                            data_id, prod_error_code, prod_error_info))
                    error_channeldata = ChannelData(
                        error_code=ChannelDataErrCode.PRODUCT_ERROR.value,
                        error_info="",
                        prod_error_code=prod_error_code,
                        prod_error_info=prod_error_info,
                        data_id=data_id,
                        log_id=log_id)
            else:
                error_channeldata = ChannelData(
                    error_code=resp.error_no,
                    error_info=resp.error_msg,
                    data_id=data_id,
                    log_id=log_id)
                skip_process_dict[data_id] = True

            if error_channeldata is not None:
                err_channeldata_dict[data_id] = error_channeldata
            else:
                preped_data_dict[data_id] = preped_data
        _LOGGER.debug("{} Succ preprocess".format(op_info_prefix))
        return preped_data_dict, err_channeldata_dict, skip_process_dict

    def _run_process(self, preped_data_dict, op_info_prefix, skip_process_dict, logid_dict):
        """
        执行 process 步骤

        Args:
            preped_data_dict: 模型要预测的数据
            op_info_prefix: prefix op info
            skip_process_dict: 是否跳过 process 步骤
            logid_dict: logid dict

        Returns:
            midped_data_dict: data midprocessed, to be post-processed
            err_channeldata_dict: 异常发生时，将其放入此
        """

        _LOGGER.debug("{} Running process".format(op_info_prefix))
        midped_data_dict = collections.OrderedDict()
        err_channeldata_dict = collections.OrderedDict()
        is_skip_process = False
        data_ids = list(preped_data_dict.keys())

        # skip process stage
        if len(data_ids) == 1 and skip_process_dict.get(data_ids[0]) is True:
            is_skip_process = True

        if self.with_serving is False or is_skip_process is True:
            midped_data_dict = preped_data_dict
            _LOGGER.warning(
                "(data_id={} log_id={}) OP={} skip process stage. with_serving={}, is_skip_process={}".format(
                    data_ids[0], logid_dict.get(data_ids[0]), self.name, self.with_serving, is_skip_process))
            return midped_data_dict, err_channeldata_dict

        # use typical_logid to mark batch data
        # data_ids is one self-increasing unique key.
        typical_logid = data_ids[0]
        if len(data_ids) != 1:
            for data_id in data_ids:
                _LOGGER.info(
                    "(data_id={} logid={}) Auto-batching is On Op={}!!"
                    "We selected logid={} (from batch: {}) as a "
                    "representative for logging.".format(
                        data_id, logid_dict.get(data_id), self.name, typical_logid, data_ids))

        one_input = preped_data_dict[data_ids[0]]
        feed_batch = []
        feed_dict = {}
        cur_offset = 0
        input_offset_dict = {}
        batch_input = False

        if isinstance(one_input, dict):
            # 对于 dict 类型，数据结构是 dict
            # 将 data_ids 中多个 dicts 合并到一个 dict
            # feed_batch 是预测函数的输入参数
            # input_offset_dict 用于 restration[data_ids]
            if len(data_ids) == 1:
                feed_batch = [preped_data_dict[data_id] for data_id in data_ids]
            else:
                for data_id in data_ids:
                    for key, val in preped_data_dict[data_id].items():
                        has_val = feed_dict.get(key)
                        if has_val is None:
                            feed_dict[key] = val
                            continue
                        # merge 2 np.arrray
                        if isinstance(val, np.ndarray):
                            feed_dict[key] = np.append(
                                feed_dict[key], val, axis=0)
                feed_batch.append(feed_dict)

            for data_id in data_ids:
                start = cur_offset
                for key, val in preped_data_dict[data_id].items():
                    if isinstance(val, (list, np.ndarray)):
                        cur_offset += len(val)
                    else:
                        cur_offset += 1
                    break
                input_offset_dict[data_id] = [start, cur_offset]
        elif isinstance(one_input, list):
            # 对于 list 类型，one_input 的数据结构是 [dict, dict, ...]
            # feed_batch 的数据结构是 [dict1_1, dict1_2, dict2_1, ...]
            # input_offset_dict 的数据结构是 { data_id: [start, end] }
            batch_input = True
            for data_id in data_ids:
                feed_batch.extend(preped_data_dict[data_id])
                data_size = len(preped_data_dict[data_id])
                start = cur_offset
                cur_offset = start + data_size
                input_offset_dict[data_id] = [start, cur_offset]
        else:
            _LOGGER.critical(
                "(data_id={} log_id={}){} Failed to process: expect input type is dict"
                " or list(batch input), but get {}".format(
                    data_ids[0], typical_logid, op_info_prefix, type(one_input)))
            for data_id in data_ids:
                error_code = ChannelDataErrCode.TYPE_ERROR.value
                error_info = "expect input type is dict or list, but get {}".format(type(one_input))
                err_channeldata_dict[data_id] = ChannelData(
                    error_code=error_code,
                    error_info=error_info,
                    data_id=data_id,
                    log_id=logid_dict.get(data_id))
            return midped_data_dict, err_channeldata_dict

        midped_batch = None
        error_code = ChannelDataErrCode.OK.value
        error_info = ""
        if self._timeout <= 0:
            # No retry
            try:
                if batch_input is False:
                    midped_batch, error_code, error_info = self.process(feed_batch, typical_logid)
                else:
                    midped_batch = []
                    for idx in range(len(feed_batch)):
                        predict_res, error_code, error_info = self.process(
                            [feed_batch[idx]], typical_logid)
                        if error_code != ChannelDataErrCode.OK.value:
                            break
                        midped_batch.append(predict_res)
            except Exception as e:
                error_code = ChannelDataErrCode.UNKNOWN.value
                error_info = "(data_id={} log_id={}) {} Failed to process(batch: {}): {}".format(
                    data_ids[0], typical_logid, op_info_prefix, data_ids, e)
                _LOGGER.error(error_info, exc_info=True)
        else:
            # retry N times configured in yaml files.
            for i in range(self._retry):
                try:
                    # time out for each process
                    if batch_input is False:
                        midped_batch, error_code, error_info = func_timeout.func_timeout(
                            self._timeout,
                            self.process,
                            args=(feed_batch, typical_logid))
                    else:
                        midped_batch = []
                        for idx in range(len(feed_batch)):
                            predict_res, error_code, error_info = func_timeout.func_timeout(
                                self._timeout,
                                self.process,
                                args=([feed_batch[idx]], typical_logid))
                            midped_batch[idx].append(predict_res)
                except func_timeout.FunctionTimedOut as e:
                    if i + 1 >= self._retry:
                        error_code = ChannelDataErrCode.TIMEOUT.value
                        error_info = "(log_id={}) {} Failed to process(batch: {}): exceeded retry count.".format(
                            typical_logid, op_info_prefix, data_ids)
                        _LOGGER.error(error_info)
                    else:
                        _LOGGER.warning(
                            "(log_id={}) {} Failed to process(batch: {}): timeout, and retrying({}/{})...".format(
                                typical_logid, op_info_prefix, data_ids, i + 1, self._retry))
                except Exception as e:
                    error_code = ChannelDataErrCode.UNKNOWN.value
                    error_info = "(log_id={}) {} Failed to process(batch: {}): {}".format(
                        typical_logid, op_info_prefix, data_ids, e)
                    _LOGGER.error(error_info, exc_info=True)
                    break
                else:
                    break

        # 2 kinds of errors
        if error_code != ChannelDataErrCode.OK.value or midped_batch is None:
            error_info = "[{}] failed to predict. {}. Please check the input dict and checkout " \
                         "PipelineServingLogs/pipeline.log for more details.".format(self.name, error_info)
            _LOGGER.error(error_info)
            for data_id in data_ids:
                err_channeldata_dict[data_id] = ChannelData(
                    error_code=error_code,
                    error_info=error_info,
                    data_id=data_id,
                    log_id=logid_dict.get(data_id))
            return midped_data_dict, err_channeldata_dict

        # 将推理结果分给每个data_ids
        if batch_input is False:
            var_names = midped_batch.keys()
            lod_var_names = set()
            lod_offset_names = set()
            # 对于单输入 midped_batch 是 dict 类型
            for name in var_names:
                lod_offset_name = "{}.lod".format(name)
                if lod_offset_name in var_names:
                    _LOGGER.debug("(log_id={}) {} {} is LodTensor".format(typical_logid, op_info_prefix, name))
                    lod_var_names.add(name)
                    lod_offset_names.add(lod_offset_name)

            for idx, data_id in enumerate(data_ids):
                midped_data_dict[data_id] = {}

            for name, value in midped_batch.items():
                if name in lod_offset_names:
                    continue
                if name in lod_var_names:
                    # lodtensor
                    lod_offset_name = "{}.lod".format(name)
                    lod_offset = midped_batch[lod_offset_name]
                    for idx, data_id in enumerate(data_ids):
                        data_offset_left = input_offset_dict[data_id][0]
                        data_offset_right = input_offset_dict[data_id][1]
                        lod_offset_left = lod_offset[data_offset_left]
                        lod_offset_right = lod_offset[data_offset_right]
                        midped_data_dict[data_id][name] = value[lod_offset_left:lod_offset_right]
                        midped_data_dict[data_id][lod_offset_name] = \
                            lod_offset[data_offset_left:data_offset_right + 1] - lod_offset[data_offset_left]
                else:
                    # normal tensor
                    for idx, data_id in enumerate(data_ids):
                        start = input_offset_dict[data_id][0]
                        end = input_offset_dict[data_id][1]
                        midped_data_dict[data_id][name] = value[start:end]
        else:
            for idx, data_id in enumerate(data_ids):
                start = input_offset_dict[data_id][0]
                end = input_offset_dict[data_id][1]
                midped_data_dict[data_id] = midped_batch[start:end]
        return midped_data_dict, err_channeldata_dict

    def _run_postprocess(self, parsed_data_dict, midped_data_dict, op_info_prefix, logid_dict):
        """
        执行 postprocess 步骤

        Args:
            parsed_data_dict: preprocess 步骤返回的数据
            midped_data_dict: process 步骤返回的数据
            op_info_prefix: prefix op info
            logid_dict: logid dict

        Returns:
            postped_data_dict: postprocess 处理后的数据
            err_channeldata_dict: 异常发生时，将其放入此 dict

        """
        _LOGGER.debug("{} Running postprocess".format(op_info_prefix))
        postped_data_dict = collections.OrderedDict()
        err_channeldata_dict = collections.OrderedDict()

        @ErrorCatch
        def postprocess_help(self, parsed_data_dict, midped_data, data_id, logid_dict):
            postped_data, prod_errcode, prod_errinfo = self.postprocess(
                parsed_data_dict[data_id], midped_data, data_id, logid_dict.get(data_id))
            if not isinstance(postped_data, dict):
                raise CustomException(CustomExceptionCode.TYPE_ERROR, "postprocess should return dict", True)
            return postped_data, prod_errcode, prod_errinfo

        for data_id, midped_data in midped_data_dict.items():
            log_id = logid_dict.get(data_id)
            postped_data, err_channeldata = None, None
            prod_errcode, prod_errinfo = None, None
            post_res, resp = postprocess_help(
                self, parsed_data_dict, midped_data, data_id=data_id, logid_dict=logid_dict)

            if resp.error_no == CustomExceptionCode.OK.value:
                postped_data, prod_errcode, prod_errinfo = post_res
                if prod_errcode is not None:
                    # 业务错误发生
                    err_channeldata = ChannelData(
                        error_code=ChannelDataErrCode.PRODUCT_ERROR.value,
                        error_info="",
                        prod_error_code=prod_errcode,
                        prod_error_info=prod_errinfo,
                        data_id=data_id,
                        log_id=log_id)
            else:
                err_channeldata = ChannelData(
                    error_code=resp.error_no,
                    error_info=resp.error_msg,
                    data_id=data_id,
                    log_id=log_id)

            if err_channeldata is not None:
                err_channeldata_dict[data_id] = err_channeldata
                continue

            output_data = None
            err, _ = ChannelData.check_npdata(postped_data)
            if err == 0:
                output_data = ChannelData(
                    ChannelDataType.CHANNEL_NPDATA.value,
                    npdata=postped_data,
                    data_id=data_id,
                    log_id=log_id)
            else:
                output_data = ChannelData(
                    ChannelDataType.DICT.value,
                    dictdata=postped_data,
                    data_id=data_id,
                    log_id=log_id)
            postped_data_dict[data_id] = output_data

        _LOGGER.debug("{} Succ postprocess".format(op_info_prefix))
        return postped_data_dict, err_channeldata_dict

    def _auto_batching_generator(self, input_channel, op_name, batch_size, timeout, op_info_prefix):
        """
        合并批量请求预测一次。每次从输入 channel 中取一份数据，
        直到 batch 大小等于 batch_size，或者等待时间超过 auto_batching_timeout

        Args:
            input_channel: Op 的输入 channel
            op_name: op name
            batch_size: 批量大小，小于 worker_num
            timeout: 批量超时时间 (秒)，如果超时时间为 None，并且从 front() 中获取的数量小于 batch_size，阻塞发生
            op_info_prefix: op 信息.

        Returns:
            None
        """
        while True:
            batch = []
            while len(batch) == 0:
                endtime = None
                if timeout is not None:
                    endtime = _time() + timeout
                for idx in range(batch_size):
                    try:
                        channeldata_dict = None
                        front_start_time = _timestamp()
                        if timeout is not None:
                            remaining = endtime - _time()
                            if remaining <= 0.0:
                                _LOGGER.debug("{} Failed to generate batch: timeout".format(op_info_prefix))
                                break
                            channeldata_dict = input_channel.front(op_name, timeout)
                        else:
                            channeldata_dict = input_channel.front(op_name)
                        batch.append(channeldata_dict)
                        _LOGGER.debug(
                            "_auto_batching_generator get {} channeldata from op:{} input channel. time={}".format(
                                idx, op_name, front_start_time))
                    except ChannelTimeoutError:
                        _LOGGER.debug("{} Failed to generate batch: timeout".format(op_info_prefix))
                        break
            _LOGGER.debug("{} Got actual batch_size: {}".format(op_info_prefix, len(batch)))
            yield batch

    def _parse_channeldata_batch(self, batch, output_channels):
        """
        解析批量 channeldatas

        Args:
            batch: 自动批量的批量数据
            output_channels: 输出 channels

        Returns:
            parsed_data_dict: parsed from channeldata in batch
            need_profile_dict: need profile dict in batch
            profile_dict: profile info dict in batch
            logid_dict: trace each request in batch
        """
        parsed_data_dict = collections.OrderedDict()
        need_profile_dict = {}
        profile_dict = {}
        logid_dict = {}
        for channeldata_dict in batch:
            data_id, error_channeldata, parsed_data, client_need_profile, profile_set, log_id = \
                self._parse_channeldata(channeldata_dict)
            if error_channeldata is None:
                parsed_data_dict[data_id] = parsed_data
                need_profile_dict[data_id] = client_need_profile
                profile_dict[data_id] = profile_set
                logid_dict[data_id] = log_id
            else:
                # error data in predecessor Op
                # (error_channeldata with profile info)
                self._push_to_output_channels(error_channeldata,
                                              output_channels)
        return parsed_data_dict, need_profile_dict, profile_dict, logid_dict

    def _run(self, concurrency_idx, input_channel, output_channels,
             is_thread_op, trace_buffer, model_config, workdir, thread_num,
             device_type, devices, mem_optim, ir_optim, precision,
             use_mkldnn, mkldnn_cache_capacity, mkldnn_op_list,
             mkldnn_bf16_op_list, is_jump_op, output_channels_of_jump_ops,
             min_subgraph_size, dynamic_shape_info, use_calib):
        """
        _run 是线程/进程 Op 的入口函数。进程模式下客户端类型为 local_predictor 时，CUDA 环境
        需要由 LocalServiceHandler[child process] 初始化，否则，CUDA 错误(3)，初始化错误。
        预处理、处理和后处理在 main 循环中执行。预处理和后处理通常由用户重写。跟踪数据由 trace_que 记录。

        Args:
            concurrency_idx: thread/process index
            input_channel: 输入 channel, 获取要预处理的数据
            output_channels: 输出 channel, 存储处理后数据
            is_thread_op: False, 进程 op; True, 线程 op
            trace_buffer: 存储跟踪信息
            model_config: 模型配置路径
            workdir: 工作目录
            thread_num: 线程数，并发量
            device_type: 设备类型，支持多种类型
            devices: gpu id 列表，默认 [cpu]
            mem_optim: 使用内存/图内存优化，默认 True
            ir_optim: 使用计算图表优化，默认 False
            precision: 推理精度. 如 "fp32", "fp16", "int8", "bf16"
            use_mkldnn: 使用 mkldnn，默认 False
            mkldnn_cache_capacity: mkldnn 缓存容量, 0 表示不限制.
            mkldnn_op_list: mkldnn 优化的 OP 列表，默认 None
            mkldnn_bf16_op_list: mkldnn bf16 优化的 OP 列表，默认 None
            is_jump_op: Op 是否由 jump op，默认 False
            output_channels_of_jump_ops: all output channels of jump ops.
            use_calib: 使用 calib 模式推理，默认 False

        Returns:
            None
        """
        _LOGGER.debug(f"[Op] {self.name}  _run() start")
        op_info_prefix = "[{}|{}]".format(self.name, concurrency_idx)

        # init ops
        profiler = None

        @ErrorCatch
        def check_helper(self: Op, is_thread_op, model_config, workdir,
                         thread_num, device_type, devices, mem_optim, ir_optim,
                         precision, use_mkldnn, mkldnn_cache_capacity, mkldnn_op_list,
                         mkldnn_bf16_op_list, min_subgraph_size, dynamic_shape_info):
            if not is_thread_op and self.client_type == "local_predictor":
                self.service_handler = local_service_handler.LocalServiceHandler(
                    model_config=model_config,
                    client_type="local_predictor",
                    workdir=workdir,
                    thread_num=thread_num,
                    device_type=device_type,
                    devices=devices,
                    mem_optim=mem_optim,
                    ir_optim=ir_optim,
                    precision=precision,
                    use_mkldnn=use_mkldnn,
                    mkldnn_cache_capacity=mkldnn_cache_capacity,
                    mkldnn_op_list=mkldnn_op_list,
                    mkldnn_bf16_op_list=mkldnn_bf16_op_list,
                    min_subgraph_size=min_subgraph_size,
                    dynamic_shape_info=dynamic_shape_info,
                    use_calib=use_calib)

                _LOGGER.info("Init cuda env in process {}".format(concurrency_idx))
                self.local_predictor = self.service_handler.get_client(concurrency_idx)
            # check all ops initialized successfully.
            profiler = self._initialize(is_thread_op, concurrency_idx)
            return profiler

        profiler, resp = check_helper(self, is_thread_op, model_config, workdir,
                                      thread_num, device_type, devices, mem_optim, ir_optim,
                                      precision, use_mkldnn, mkldnn_cache_capacity, mkldnn_op_list,
                                      mkldnn_bf16_op_list, min_subgraph_size, dynamic_shape_info)

        if resp.error_no != CustomExceptionCode.OK.value:
            _LOGGER.critical("{} failed to init op: {}".format(op_info_prefix, resp.error_msg), exc_info=False)
            print("{} failed to init op: {}".format(op_info_prefix, resp.error_msg))
            kill_stop_process_by_pid("kill", os.getpgid(os.getpid()))

        _LOGGER.info("{} Succ init".format(op_info_prefix))

        batch_generator = self._auto_batching_generator(
            input_channel=input_channel,
            op_name=self.name,
            batch_size=self._batch_size,
            timeout=self._auto_batching_timeout,
            op_info_prefix=op_info_prefix)

        start, end = None, None
        trace_que = collections.deque()
        while True:
            start = _timestamp()
            try:
                channeldata_dict_batch = next(batch_generator)
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break
            end = _timestamp()
            in_time = end - start
            _LOGGER.debug("op:{} in_time_end:{}".format(op_info_prefix, time.time()))

            # parse channeldata batch
            try:
                parsed_data_dict, need_profile_dict, profile_dict, logid_dict = self._parse_channeldata_batch(
                    channeldata_dict_batch, output_channels)
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break

            if len(parsed_data_dict) == 0:
                # 整个批量都是错误数据
                continue
            _LOGGER.debug("op:{} parse_end:{}".format(op_info_prefix, time.time()))

            front_cost = _timestamp()
            for data_id, parsed_data in parsed_data_dict.items():
                _LOGGER.debug("(data_id={}) POP INPUT CHANNEL! op:{}, cost:{} ms".format(
                    data_id, self.name, front_cost / 1000.0))

            # preprecess
            start = profiler.record("prep#{}_0".format(op_info_prefix))
            preped_data_dict, err_channeldata_dict, skip_process_dict = self._run_preprocess(
                parsed_data_dict, op_info_prefix, logid_dict)
            end = profiler.record("prep#{}_1".format(op_info_prefix))
            prep_time = end - start
            _LOGGER.debug("op:{} preprocess_end:{}, cost:{} ms".format(op_info_prefix, time.time(), prep_time / 1000.0))

            try:
                # 将错误请求放到输出 channel，跳过 process 和 postprocess 步骤
                for data_id, err_channeldata in err_channeldata_dict.items():
                    self._push_to_output_channels(
                        data=err_channeldata,
                        channels=output_channels,
                        client_need_profile=need_profile_dict[data_id],
                        profile_set=profile_dict[data_id])
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break
            if len(preped_data_dict) == 0:
                continue

            # process
            start = profiler.record("midp#{}_0".format(op_info_prefix))
            midped_data_dict, err_channeldata_dict = self._run_process(
                preped_data_dict, op_info_prefix, skip_process_dict, logid_dict)
            end = profiler.record("midp#{}_1".format(op_info_prefix))
            _LOGGER.info("prometheus inf count +1")
            midp_time = end - start
            _LOGGER.debug("op:{} process_end:{}, cost:{} ms".format(
                op_info_prefix, time.time(), midp_time / 1000.0))
            try:
                for data_id, err_channeldata in err_channeldata_dict.items():
                    self._push_to_output_channels(
                        data=err_channeldata,
                        channels=output_channels,
                        client_need_profile=need_profile_dict[data_id],
                        profile_set=profile_dict[data_id])
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break
            if len(midped_data_dict) == 0:
                continue

            # postprocess
            start = profiler.record("postp#{}_0".format(op_info_prefix))
            postped_data_dict, err_channeldata_dict = self._run_postprocess(
                parsed_data_dict, midped_data_dict, op_info_prefix, logid_dict)
            end = profiler.record("postp#{}_1".format(op_info_prefix))
            postp_time = end - start
            after_postp_time = _time()
            _LOGGER.debug("op:{} postprocess_end:{}, cost:{} ms".format(
                op_info_prefix, time.time(), postp_time / 1000.0))
            try:
                for data_id, err_channeldata in err_channeldata_dict.items():
                    self._push_to_output_channels(
                        data=err_channeldata,
                        channels=output_channels,
                        client_need_profile=need_profile_dict[data_id],
                        profile_set=profile_dict[data_id])
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break
            if len(postped_data_dict) == 0:
                continue

            # push data to channel (if run succ)
            start = _timestamp()
            try:
                profile_str = profiler.gen_profile_str()
                if self.is_jump_op() is True and self.check_jumping(postped_data_dict) is True:
                    # push data to output channel of ops to be jumped
                    for data_id, postped_data in postped_data_dict.items():
                        if self._server_use_profile:
                            sys.stderr.write(profile_str)
                        self._push_to_output_channels(
                            data=postped_data,
                            channels=output_channels_of_jump_ops,
                            profile_str=profile_str,
                            client_need_profile=need_profile_dict[data_id],
                            profile_set=profile_dict[data_id])
                        after_outchannel_time = _time()
                        _LOGGER.debug(
                            "(data_id={}) PUSH OUTPUT CHANNEL OF JUMP OPs! op:{} push cost:{} ms".format(
                                data_id, self.name, (after_outchannel_time - after_postp_time) * 1000))
                else:
                    # push data to output channel.
                    for data_id, postped_data in postped_data_dict.items():
                        if self._server_use_profile:
                            sys.stderr.write(profile_str)
                        self._push_to_output_channels(
                            data=postped_data,
                            channels=output_channels,
                            profile_str=profile_str,
                            client_need_profile=need_profile_dict[data_id],
                            profile_set=profile_dict[data_id])
                        after_outchannel_time = _time()
                        _LOGGER.debug(
                            "(data_id={}) PUSH OUTPUT CHANNEL! op:{} push cost:{} ms".format(
                                data_id, self.name, (after_outchannel_time - after_postp_time) * 1000))
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break

            end = _timestamp()
            out_time = end - start
            after_outchannel_time = _timestamp()
            if trace_buffer is not None:
                trace_que.append({
                    "name": self.name,
                    "actions": {
                        "in": in_time,
                        "prep": prep_time,
                        "midp": midp_time,
                        "postp": postp_time,
                        "out": out_time,
                    }
                })
                while trace_que:
                    info = trace_que[0]
                    try:
                        trace_buffer.put_nowait(info)
                        trace_que.popleft()
                    except Queue.Full:
                        break
        _LOGGER.debug(f"[Op] {self.name}  _run() end")

    def _initialize(self, is_thread_op, concurrency_idx):
        """
        在线程或进程中目标函数中初始化一个 Op 对象。
        使用 _client_config 和 _server_endpoints 初始化客户端对象。
        为每个线程或进程创建一个 TimeProfiler 来记录分析器信息。

        Args:
            is_thread_op: True 线程 Op，False 进程 Op。
            concurrency_idx: 进程 id, 线程模式不用此参数.

        Returns:
            TimeProfiler
        """
        _LOGGER.debug(f"[Op] {self.name} _initialize() start")

        @ErrorCatch
        def init_helper(self, is_thread_op, concurrency_idx):
            if is_thread_op:
                with self._for_init_op_lock:
                    if not self._succ_init_op:
                        # 对于线程 Op，每个线程不能得到它自己的 concurrency_idx
                        self.concurrency_idx = None
                        # init client
                        self.client = self.init_client(self._client_config, self._server_endpoints)
                        # user defined
                        self.init_op()
                        self._succ_init_op = True
                        self._succ_close_op = False
            else:
                self.concurrency_idx = concurrency_idx
                # init client
                self.client = self.init_client(self._client_config, self._server_endpoints)
                # user defined
                self.init_op()

        init_helper(self, is_thread_op, concurrency_idx)
        # 每个进程或线程独立一个 TimeProfiler
        profiler = TimeProfiler()
        profiler.enable(True)
        _LOGGER.debug(f"[Op] {self.name} _initialize() end")
        return profiler

    def _finalize(self, is_thread_op):
        _LOGGER.debug(f"[Op] {self.name} _finalize() start")
        if is_thread_op:
            with self._for_close_op_lock:
                if not self._succ_close_op:
                    self._profiler = None
                    self.client = None
                    self._succ_init_op = False
                    self._succ_close_op = True
        _LOGGER.debug(f"[Op] {self.name} _finalize() end")

    def _log(self, info):
        """
        日志格式化：Op_name + info
        """
        return "{} {}".format(self.name, info)


class RequestOp(Op):
    """
    请求 Op，用于 unpacking 一个请求包裹。如果需要特殊的 unpacking 方法，
    你需要继承 RequestOp 类，重写 unpack_request_package 方法。
    注意！！！ RequestOp 不执行 preprocess, process, postprocess.
    """

    def __init__(self):
        # PipelineService.name = "@DAGExecutor"
        super(RequestOp, self).__init__(name="@DAGExecutor", input_ops=[])
        _LOGGER.debug(f"[RequestOp] init one RequestOp name={self.name}")
        # init op
        try:
            self.init_op()
        except Exception as e:
            _LOGGER.critical("Op(Request) Failed to init: {}".format(e))
            os._exit(-1)

    def proto_tensor_2_numpy(self, tensor):
        """
        Convert proto tensor to numpy array, The supported types are as follows:
                INT64
                FP32
        INT32
        FP64
        INT16
        FP16
        BF16
        UINT8
        INT8
        BOOL
                BYTES
        Unsupported type:
                STRING
                COMPLEX64
                COMPLEX128

        Args:
            tensor: one tensor in request.tensors.

        Returns:
            np_data: np.ndnumpy, the tensor data is converted to numpy.
            lod_info: np.ndnumpy, lod info of the tensor data, None default.
        """
        pass

    def unpack_request_package(self, request):
        """
        Unpack request package by gateway.proto
        Args:
            request: HTTP body, JSON format

        Returns:
            dict_data: json fields in HTTP body
            log_id: log_id
            prod_errcode: None or ProductErrCode.SUCC.value default, otherwise,
                          product errores occured.It is handled in the same way
                          as exception.
            prod_errinfo: "" default
        """
        _LOGGER.debug("[RequestOp] unpack_request_package() start()")
        dict_data = {}
        log_id = None
        if request is None:
            _LOGGER.critical("request is None")
            raise ValueError("request is None")

        # unpack key/value string list
        for idx, key in enumerate(request.key):
            dict_data[key] = request.value[idx]
        log_id = request.logid

        # unpack proto.tensors data.


        _LOGGER.info("RequestOp unpack one request. log_id:{}, clientip:{}, time:{}".format(
            log_id, request.clientip, time.time()))

        return dict_data, log_id, None, ""


class ResponseOp(Op):
    """
    响应 Op，用于 packing 一个响应包裹。如果需要特殊的 packing 方法，
    你需要继承 Response 类，重写 pack_response_package 方法。
    注意！！！ResponseOp 不执行 不执行 preprocess, process, postprocess.
    """

    def __init__(self, input_ops):
        super(ResponseOp, self).__init__(name="@DAGExecutor", input_ops=input_ops)
        _LOGGER.debug(f"[ResponseOp] init one ResponseOp name={self.name}")
        # init op
        try:
            self.init_op()
        except Exception as e:
            _LOGGER.critical("Op(ResponseOp) Failed to init: {}".format(e, exc_info=True))
            os._exit(-1)

        # init ResponseOp
        self.is_pack_tensor = False

    def set_pack_format(self, isTensor=False):
        self.is_pack_tensor = isTensor

    def pack_response_package(self, channeldata: ChannelData):
        """
        从最后的 channel 中获取 channeldata，打包 protobuf 序列化后响应包裹

        Args:
            channeldata: Type ChannelData

        Returns:
            resp: pipeline_service_pb2.Response()
        """
        resp = pipeline_service_pb2.Response()
        error_code = channeldata.error_code
        error_info = ""
        if error_code == ChannelDataErrCode.OK.value:
            # 框架级别错误
            if channeldata.datatype == ChannelDataType.CHANNEL_NPDATA.value:
                feed = channeldata.parse()
                # ndarray to string:
                # https://stackoverflow.com/questions/30167538/convert-a-numpy-ndarray-to-stringor-bytes-and-convert-it-back-to-numpy-ndarray
                np.set_printoptions(threshold=sys.maxsize)
                for name, var in feed.items():
                    resp.value.append(var.__repr__())
                    resp.key.append(name)
            elif channeldata.datatype == ChannelDataType.DICT.value:
                feed = channeldata.parse()
                for name, var in feed.items():
                    # if not isinstance(var, str):
                    #     error_code = ChannelDataErrCode.TYPE_ERROR.value
                    #     error_info = self._log("fetch var type must be str({}).".format(type(var)))
                    #     _LOGGER.error("(logid={}) Failed to pack RPC response package: {}".format(
                    #         channeldata.id, error_info))
                    #     break
                    resp.value.append(var)
                    resp.key.append(name)
            else:
                error_code = ChannelDataErrCode.TYPE_ERROR.value
                error_info = self._log("error type({}) in datatype.".format(channeldata.datatype))
                _LOGGER.error("(logid={}) Failed to pack RPC response package: {}".format(channeldata.id, error_info))
        else:
            # Product level errors
            error_info = channeldata.error_info
            if error_code == ChannelDataErrCode.PRODUCT_ERROR.value:
                # rewrite error_code when product errors occured
                error_code = channeldata.prod_error_code
                error_info = channeldata.prod_error_info

        # pack results
        if error_code is None:
            error_code = 0
        resp.error_no = error_code
        resp.error_msg = error_info

        return resp


class VirtualOp(Op):
    """
    为了在 dag 视图中跨级连接两个 ops，在非虚拟 Ops 间创建虚拟 ops，并且只传输数据。
    例如，F 的前驱 Ops 是 D 和 E。在构建 DAG 时，根据 dag 视图逐层创建 channel 层。
    Op F 不在 [B, E] 下一层视图，因此创建一个虚拟 Op 'V1' ，它的前驱 Op 是 E，等等，
    创建两个虚拟 Op 'V2' 和 'V3'，最终，找到了非虚拟 Op F。在 E、V1、V2、V3 和 F 中
    创建了 4 个 channels，V1、V2、V3 和 F 的生产者是 E。

        DAG: [A -> B -> C -> D -> F]
               \-> E ----------/

        DAG view: [[A], [B, E], [C], [D], [F]]
        BUILD DAG: [A -> B -> C -> D -> E -> F]
                     \-> E -> V1-> V2-> V3/
    """

    def __init__(self, name, concurrency=1):
        super(VirtualOp, self).__init__(name=name, input_ops=None, concurrency=concurrency)
        _LOGGER.debug(f"[VirtualOp] init one VirtualOp name={self.name}")
        self._virtual_pred_ops = []

    def add_virtual_pred_op(self, op):
        """
        添加当前虚拟 Op 的前驱 Op

        Args:
            op: Op 对象，可能是虚拟的，也可能不是

        Returns:
            None
        """
        self._virtual_pred_ops.append(op)

    def _actual_pred_op_names(self, op):
        """
        递归地查找前驱 Op 中的非虚拟 Op，找到任意一个就返回

        Args:
            op: Op 对象

        Returns:
            names: 前驱 Op 中非虚拟 Op 的名字
        """
        # can use disjoint-set, but it's not necessary
        if not isinstance(op, VirtualOp):
            return [op.name]
        names = []
        for x in op._virtual_pred_ops:
            names.extend(self._actual_pred_op_names(x))
        return names

    def add_output_channel(self, channel):
        """
        Adding the output channel of non-virtual pred ops.

        Args:
            channel: one channel.

        Returns:
            None.
        """
        if not isinstance(channel, (ThreadChannel, ProcessChannel)):
            _LOGGER.critical(self._log(
                "Failed to add output_channel: output_channel must be Channel type, not {}".format(type(channel))))
            os._exit(-1)
        for op in self._virtual_pred_ops:
            for op_name in self._actual_pred_op_names(op):
                channel.add_producer(op_name)
        self._outputs.append(channel)

    def _run(self, concurrency_idx, input_channel, output_channels, client_type, is_thread_op):
        """
        函数 _run() 只在一个线程/进程中的 Ops 之间传输数据

        Args:
            concurrency_idx: 进程 id，线程模式不可用
            input_channel: 输入 channel
            output_channels: 输出 channels
            client_type: 无用
            is_thread_op: True, 线程模式; False, 进程模式

        Returns:
            None
        """
        op_info_prefix = "[{}|{}]".format(self.name, concurrency_idx)
        # log = get_log_func(op_info_prefix)
        log = op_info_prefix
        tid = threading.current_thread().ident

        batch_generator = self._auto_batching_generator(
            input_channel=input_channel, op_name=self.name, batch_size=1, timeout=None, op_info_prefix=log)

        while True:
            try:
                channeldata_dict_batch = next(batch_generator)
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break

            try:
                for channeldata_dict in channeldata_dict_batch:
                    for name, data in channeldata_dict.items():
                        self._push_to_output_channels(data, channels=output_channels, name=name)
            except ChannelStopError:
                _LOGGER.debug("{} Stop.".format(op_info_prefix))
                self._finalize(is_thread_op)
                break
