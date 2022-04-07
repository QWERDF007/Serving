import os
import multiprocessing
from pipeline import util
from utils.logger import get_logger, set_logger_level

_LOGGER = get_logger()
_workdir_name_gen = util.NameGenerator("workdir_")


class LocalServiceHandler(object):
    """
    LocalServiceHandler 是本地服务的处理器，包括两种类型，grpc 和 local_predictor。如果使用 grpc，
    则提供服务启动能力。如果使用 local_predictor，本地预测是由 ?? 提供的。
    """

    def __init__(self,
                 model_config,
                 client_type='local_predictor',
                 workdir="",
                 thread_num=2,
                 device_type=-1,
                 devices="",
                 fetch_names=None,
                 mem_optim=True,
                 ir_optim=False,
                 available_port_generator=None,
                 use_profile=False,
                 precision="fp32",
                 use_mkldnn=False,
                 mkldnn_cache_capacity=0,
                 mkldnn_op_list=None,
                 mkldnn_bf16_op_list=None,
                 min_subgraph_size=3,
                 dynamic_shape_info={},
                 use_calib=False):
        """

        Args:
            model_config: 模型配置路径
            client_type: grpc 和 local_predictor (默认)
            workdir: 工作目录
            thread_num: 线程数，并发量
            device_type: -1：由 `devices` 决定；0：cpu；1：gpu；2：tensorRT；3：arm cpu；4：kunlun xpu
            devices: gpu id 列表。""：默认 cpu
            fetch_names:
            mem_optim: 内存/图内存优化，默认 True
            ir_optim: 计算图表优化，默认 False
            available_port_generator: 生成可用的端口
            use_profile: 性能分析，默认 False
            precision: 推理精度，默认 fp32
            use_mkldnn: 使用 mkldnn，默认 False
            mkldnn_cache_capacity: mkldnn 缓存容量，0 意味不限制
            mkldnn_op_list: mkldnn优化后的 Op，默认 None
            mkldnn_bf16_op_list: mkldnn bf16 优化后的 Op 列表，默认 None
            min_subgraph_size:
            dynamic_shape_info:
            use_calib: 设置推理 `use_calib_mode` 参数，默认 False
        """
        _LOGGER.debug("[LocalServiceHandler] init LocalServiceHandler start")
        if available_port_generator is None:
            available_port_generator = util.GetAvailablePortGenerator()

        self._model_config = model_config
        self._port_list = []
        self._device_name = "cpu"
        self._use_gpu = False
        self._use_trt = False
        self._use_lite = False
        self._use_xpu = False
        self._use_ascend_cl = False
        self._use_mkldnn = False
        self._mkldnn_cache_capacity = 0
        self._mkldnn_op_list = None
        self._mkldnn_bf16_op_list = None
        self.min_subgraph_size = 3
        self.dynamic_shape_info = {}
        self._use_calib = False

        if device_type == -1:
            # device_type is not set, determined by `devices`,
            if devices == "":
                # CPU
                self._device_name = "cpu"
                devices = [-1]
            else:
                # GPU
                self._device_name = "gpu"
                self._use_gpu = True
                devices = [int(x) for x in devices.split(",")]
        elif device_type == 0:
            # CPU
            self._device_name = "cpu"
            devices = [-1]
        elif device_type == 1:
            # GPU
            self._device_name = "gpu"
            self._use_gpu = True
            devices = [int(x) for x in devices.split(",")]
        elif device_type == 2:
            # Nvidia Tensor RT
            self._device_name = "gpu"
            self._use_gpu = True
            devices = [int(x) for x in devices.split(",")]
            self._use_trt = True
            self.min_subgraph_size = min_subgraph_size
            self.dynamic_shape_info = dynamic_shape_info
        elif device_type == 3:
            # ARM CPU
            self._device_name = "arm"
            devices = [-1]
            self._use_lite = True
        elif device_type == 4:
            # Kunlun XPU
            self._device_name = "arm"
            devices = [int(x) for x in devices.split(",")]
            self._use_lite = True
            self._use_xpu = True
        elif device_type == 5:
            # Ascend 310 ARM CPU
            self._device_name = "arm"
            devices = [int(x) for x in devices.split(",")]
            self._use_lite = True
            self._use_ascend_cl = True
        elif device_type == 6:
            # Ascend 910 ARM CPU
            self._device_name = "arm"
            devices = [int(x) for x in devices.split(",")]
            self._use_ascend_cl = True
        else:
            _LOGGER.error(f"LocalServiceHandler initialization fail. device_type={device_type}")

        # 类型为 grpc 时为每个设备添加一个端口？
        if client_type == "grpc":
            for _ in devices:
                self._port_list.append(available_port_generator.next())
            _LOGGER.info(f"Create ports for devices:{devices}. Port:{self._port_list}")

        self._client_type = client_type
        self._workdir = workdir
        self._devices = devices
        self._thread_num = thread_num
        self._mem_optim = mem_optim
        self._ir_optim = ir_optim
        self._local_predictor_client = None
        self._rpc_service_list = []
        self._server_pros = []
        self._use_profile = use_profile
        self._fetch_names = fetch_names
        self._precision = precision
        self._use_mkldnn = use_mkldnn
        self._mkldnn_cache_capacity = mkldnn_cache_capacity
        self._mkldnn_op_list = mkldnn_op_list
        self._mkldnn_bf16_op_list = mkldnn_bf16_op_list
        self._use_calib = use_calib

        _LOGGER.info(
            f"Models({model_config}) will be launched by device {self._device_name}. use_gpu: {self._use_gpu}, "
            f"use_trt: {self._use_trt}, use_lite: {self._use_lite}, use_xpu: {self._use_xpu}, "
            f"device_type: {device_type}, devices: {self._devices}, mem_optim: {self._mem_optim}, "
            f"ir_optim: {self._ir_optim}, use_profile: {self._use_profile}, thread_num: {self._thread_num}, "
            f"client_type: {self._client_type}, fetch_names: {self._fetch_names}, precision: {self._precision}, "
            f"use_calib: {self._use_calib}, use_mkldnn: {self._use_mkldnn}, "
            f"mkldnn_cache_capacity: {self._mkldnn_cache_capacity}, mkldnn_op_list: {self._mkldnn_op_list}, "
            f"mkldnn_bf16_op_list: {self._mkldnn_bf16_op_list}, use_ascend_cl: {self._use_ascend_cl}, "
            f"min_subgraph_size: {self.min_subgraph_size}, "
            f"is_set_dynamic_shape_info: {bool(len(self.dynamic_shape_info))}"
        )

        _LOGGER.debug("[LocalServiceHandler] init LocalServiceHandler end")

    def get_fetch_list(self):
        return self._fetch_names

    def get_port_list(self):
        return self._port_list

    def get_client(self, concurrency_idx):
        """
        仅用于 local_predictor，创建一个 LocalPredictor 对象，并使用方法 load_model_config 初始化。
        concurrency_idx 用来选择运行设备。

        Args:
            concurrency_idx: 进程/线程 index

        Returns:
            _local_predictor_client
        """

        _LOGGER.debug("[LocalServiceHandler] get_client() start")
        device_num = len(self._devices)
        if device_num <= 0:
            _LOGGER.error(f"device_num bust be greater than 0. devices: {self._devices}")
            raise ValueError("The number of self._devices error")

        if concurrency_idx < 0:
            _LOGGER.error(f"concurrency_idx({concurrency_idx}) must be one positive number")
            concurrency_idx = 0
        elif concurrency_idx >= device_num:
            concurrency_idx = concurrency_idx % device_num

        _LOGGER.info(f"GET_CLIENT: concurrency_idx={concurrency_idx}, device_num={device_num}")

        from pipeline.local_predictor import LocalPredictor
        if self._local_predictor_client is None:
            _LOGGER.debug("[LocalServiceHandler] create one LocalPredictor")
            self._local_predictor_client = LocalPredictor()
            _LOGGER.debug("[LocalServiceHandler] LocalPredictor.load_model_config()")
            self._local_predictor_client.load_model_config(
                model_path=self._model_config,
                use_gpu=self._use_gpu,
                gpu_id=self._devices[concurrency_idx],
                use_profile=self._use_profile,
                thread_num=self._thread_num,
                mem_optim=self._mem_optim,
                ir_optim=self._ir_optim,
                use_trt=self._use_trt,
                use_lite=self._use_lite,
                use_xpu=self._use_xpu,
                precision=self._precision,
                use_mkldnn=self._use_mkldnn,
                mkldnn_cache_capacity=self._mkldnn_cache_capacity,
                mkldnn_op_list=self._mkldnn_op_list,
                mkldnn_bf16_op_list=self._mkldnn_bf16_op_list,
                use_ascend_cl=self._use_ascend_cl,
                min_subgraph_size=self.min_subgraph_size,
                dynamic_shape_info=self.dynamic_shape_info,
                use_calib=self._use_calib,
            )
        _LOGGER.debug("[LocalServiceHandler] get_client() end")
        return self._local_predictor_client

    def get_client_config(self):
        return self._model_config

    def _prepare_one_server(self, workdir, port, gpuid, thread_num, mem_optim, ir_optim, precision):
        _LOGGER.warning("[LocalServiceHandler] has not supported right now")

    def _start_one_server(self, service_idx):
        """ 启动一个服务 """
        _LOGGER.debug("[LocalServiceHandler] _start_one_server()")
        self._rpc_service_list[service_idx].run_server()

    def prepare_server(self):
        """ 准备所有要开始的服务，并将它们加入列表中 """
        _LOGGER.debug("[LocalServiceHandler] prepare_server() start")
        for i, device_id in enumerate(self._devices):
            if self._workdir != "":
                workdir = f"{self._workdir}_{i}"
            else:
                workdir = _workdir_name_gen.next()
            self._rpc_service_list.append(self._prepare_one_server(
                workdir=workdir,
                port=self._port_list[i],
                gpuid=device_id,
                thread_num=self._thread_num,
                mem_optim=self._mem_optim,
                ir_optim=self._ir_optim,
                precision=self._precision
            ))
        _LOGGER.debug("[LocalServiceHandler] prepare_server() end")

    def start_server(self):
        _LOGGER.debug("[LocalServiceHandler] start_server() start")
        for i, _ in enumerate(self._rpc_service_list):
            p = multiprocessing.Process(target=self._start_one_server, args=(i,))
            p.daemon = True
            self._server_pros.append(p)
        for p in self._server_pros:
            p.start()
        _LOGGER.debug("[LocalServiceHandler] start_server() end")
