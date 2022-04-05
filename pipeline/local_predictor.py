import os
import io
import glob
import logging
import platform

import torch
import numpy as np
import yaml
import google.protobuf.text_format

from pipeline.error_catch import ErrorCatch
from inference.predictor import create_predictor
from inference.config import Config
from pipeline.error_catch import CustomExceptionCode
from utils.process_util import kill_stop_process_by_pid
from utils.logger import get_logger

_LOGGER = get_logger("D2LServing")

precision_map = {
    "int8": torch.int8,
    "int16": torch.int16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class LocalPredictor(object):
    """
    在本地环境的当前进程中预测，在进程中调用，与 RPC/HTTP 比较，
    LocalPredictor 具有更好的性能，因为没有网络和打包的负担。
    """

    def __init__(self):
        self.feed_names_ = []
        self.fetch_names_ = []
        self.feed_types_ = {}
        self.fetch_types_ = {}
        self.feed_shapes_ = {}
        self.fetch_shapes_ = {}
        self.feed_names_to_idx_ = {}
        self.fetch_names_to_idx_ = {}
        self.fetch_names_to_type_ = {}

    def search_suffix_files(self, model_path, target_suffix):
        """
        在指定目录下查找所有后缀为 target_suffix 的文件.

        Args:
            model_path: 模型目录，不能为空
            target_suffix: 目标后缀，不能为空. e.g: *.pdmodel

        Returns:
            file_list, None, [] or [path, ] .
        """
        if model_path is None or target_suffix is None:
            return None

        file_list = glob.glob(os.path.join(model_path, target_suffix))
        return file_list

    def load_model_config(self,
                          model_path,
                          use_gpu=False,
                          gpu_id=0,
                          use_profile=False,
                          thread_num=1,
                          mem_optim=True,
                          ir_optim=False,
                          use_trt=False,
                          use_lite=False,
                          use_xpu=False,
                          precision="fp32",
                          use_mkldnn=False,
                          mkldnn_cache_capacity=0,
                          mkldnn_op_list=None,
                          mkldnn_bf16_op_list=None,
                          use_feed_fetch_ops=False,
                          use_ascend_cl=False,
                          min_subgraph_size=3,
                          dynamic_shape_info={},
                          use_calib=False):
        """
        加载模型配置和创建预测器

        Args:
            model_path: 模型配置路径
            use_gpu: 使用 gpu 计算，默认 False
            gpu_id: gpu id，默认 0
            use_profile: 使用预测器性能分析，默认 False
            thread_num: cpu 数学库线程数，默认 1
            mem_optim: 内存优化，默认 True
            ir_optim: 打开计算图表优化，默认 False
            use_trt: 使用 Nvidia TensorRT 优化，默认 False
            use_lite: 使用轻量级引擎，默认 False
            use_xpu: 使用 xpu 预测，默认 False
            precision: 预测精度，默认 "fp32"
            use_mkldnn: 使用 MKLDNN，默认 False
            mkldnn_cache_capacity: 输入 shapes 的缓存容量，默认 0
            mkldnn_op_list: 使用 MKLDNN 加速的 op 列表，默认 None
            mkldnn_bf16_op_list: 使用 MKLDNN bf16 加速的 op 列表，默认 None
            use_feed_fetch_ops: 使用 feed/fetch ops，默认 False
            use_ascend_cl: 使用 Huawei Ascend 预测，默认 False
            min_subgraph_size: TensorRT 要优化的最小子图的大小，默认 3
            dynamic_shape_info: 包含 min_input_shape, max_input_shape, opt_input_shape 的字典，默认 {}
            use_calib: 使用 TensorRT 校准，默认 False

        Returns:

        """
        _LOGGER.debug(f"[LocalPredictor] load_model_config(model_path={model_path})")
        gpu_id = int(gpu_id)
        # serving/core/configure/proto/general_model_config.proto
        # 这里原本用 prototxt 文件，此处改为 yaml 文件，后面取值部分进行相应修改
        config = Config(model_path)

        _LOGGER.info(
            "LocalPredictor load_model_config params: model_path: {}, use_gpu: {}, "
            "gpu_id: {}, use_profile: {}, thread_num: {}, mem_optim: {}, ir_optim: {}, "
            "use_trt: {}, use_lite: {}, use_xpu: {}, precision: {}, use_calib: {}, "
            "use_mkldnn: {}, mkldnn_cache_capacity: {}, mkldnn_op_list: {}, "
            "mkldnn_bf16_op_list: {}, use_feed_fetch_ops: {}, "
            "use_ascend_cl: {}, min_subgraph_size: {}, dynamic_shape_info: {}".format(
                model_path, use_gpu, gpu_id, use_profile, thread_num,
                mem_optim, ir_optim, use_trt, use_lite, use_xpu, precision,
                use_calib, use_mkldnn, mkldnn_cache_capacity, mkldnn_op_list,
                mkldnn_bf16_op_list, use_feed_fetch_ops, use_ascend_cl,
                min_subgraph_size, dynamic_shape_info))

        # check precision
        precision_type = torch.float32
        if precision is not None and precision.lower() in precision_map:
            precision_type = precision_map[precision.lower()]
        else:
            _LOGGER.warning(f"precision error!!! Please check precision: {precision}")

        if use_profile:
            pass  # TODO

        if mem_optim:
            pass  # TODO

        if ir_optim:
            pass  # TODO

        if use_feed_fetch_ops:
            pass  # TODO

        # pass optim TODO

        # 设置 cpu 和 mkldnn
        if use_mkldnn:
            pass  # TODO

        # 设置 gpu
        if not use_gpu:
            config.disable_gpu()
        else:
            config.enable_use_gpu(gpu_id)
            if use_trt:
                pass  # TODO

        # 设置 lite:
        if use_lite:
            pass  # TODO

        if use_xpu:
            pass  # TODO

        if use_ascend_cl:
            if use_lite:
                pass  # TODO
            else:
                pass  # TODO

        # 设置 cpu 低精度
        if not use_gpu and not use_lite:
            if precision_type == torch.int8:
                _LOGGER.warning("precision int8 is not supported in CPU right now! Please use fp16 or bf16")
            if precision is not None and precision.lower() == "bf16":
                pass  # TODO
                if mkldnn_bf16_op_list is not None:
                    pass  # TODO

        @ErrorCatch
        def create_predictor_check(config):
            predictor = create_predictor(config)
            return predictor

        # 创建一个 predictor 实例，预测用
        predictor, resp = create_predictor_check(config)
        if resp.error_no != CustomExceptionCode.OK.value:
            _LOGGER.critical(f"failed to create predictor: {resp.error_msg}", exc_info=False)
            print(f"failed to create predictor: {resp.error_msg}")
            if platform.platform().startswith("Win"):
                kill_stop_process_by_pid("kill", os.getpid())
            else:
                kill_stop_process_by_pid("kill", os.getpgid(os.getpid()))
        self.predictor = predictor

    def predict(self, feed=None, fetch=None, batch=False, log_id=0):
        """
        模型预测

        Args:
            feed: feed 变量列表，不允许 None
            fetch: fetch 变量列表，允许 None。为 None 时，返回所有 fetch 变量，否则返回指定的结果。
            batch: 是否批量数据，默认 False。如果 batch 为 False，会在 shape 前面添加一个维度 [np.newaxis].
            log_id:

        Returns:

        """

        if feed is None:
            raise ValueError(f"You should specify feed vars for prediction. log_id: {log_id}")

        # TODO feed_batch 好像没用到
        feed_batch = []
        if isinstance(feed, dict):
            feed_batch.append(feed)
        elif isinstance(feed, list):
            feed_batch = feed
        else:
            raise ValueError(f"Feed only accepts dict and list of dict. log_id: {log_id}")

        fetch_list = []
        if fetch is not None:
            if isinstance(fetch, str):
                fetch_list = [fetch]
            elif isinstance(fetch, list):
                fetch_list = fetch

        # 过滤无效的 fetch names
        fetch_nams = []
        for key in fetch_list:
            if key in self.fetch_names_:
                fetch_nams.append(key)

        input_names = self.predictor.get_input_names()
        for name in input_names:
            if isinstance(feed[name], list) and not isinstance(feed[name][0], str):
                feed[name] = np.array(feed[name]).reshape(self.feed_shapes_[name])
            if self.feed_types_[name] == 0:
                feed[name] = feed[name].astype("int64")
            elif self.feed_types_[name] == 1:
                feed[name] = feed[name].astype("float32")
            elif self.feed_types_[name] == 2:
                feed[name] = feed[name].astype("int32")
            elif self.feed_types_[name] == 3:
                feed[name] = feed[name].astype("float64")
            elif self.feed_types_[name] == 4:
                feed[name] = feed[name].astype("int16")
            elif self.feed_types_[name] == 5:
                feed[name] = feed[name].astype("float16")
            elif self.feed_types_[name] == 6:
                feed[name] = feed[name].astype("uint16")
            elif self.feed_types_[name] == 7:
                feed[name] = feed[name].astype("uint8")
            elif self.feed_types_[name] == 8:
                feed[name] = feed[name].astype("int8")
            elif self.feed_types_[name] == 9:
                feed[name] = feed[name].astype("bool")
            elif self.feed_types_[name] == 10:
                feed[name] = feed[name].astype("complex64")
            elif self.feed_types_[name] == 11:
                feed[name] = feed[name].astype("complex128")
            elif isinstance(feed[name], list) and isinstance(feed[name][0], str):
                pass
            else:
                raise ValueError("local predictor receives wrong data tyep")

            # TODO input_tensor_handle 好像也没用到？ 还是说将tensor放到与predictor同一设备上？
            input_tensor_handle = self.predictor.get_input_handle(name)
            if "{}.lod".format(name) in feed:
                input_tensor_handle.set_lod([feed["{}.lod".format(name)]])
            if not batch:
                input_tensor_handle.copy_from_cpu(feed[name][np.newaxis, :])
            else:
                input_tensor_handle.copy_from_cpu(feed[name])

        # 设置输出 tensor handlers
        output_tensor_handles = []
        output_name_to_index_dict = {}
        output_names = self.predictor.get_output_names()
        for i, output_name in enumerate(output_names):
            output_tensor_handle = self.predictor.get_output_handle(output_name)
            output_tensor_handles.append(output_tensor_handle)
            output_name_to_index_dict[output_name] = i

        # 预测
        self.predictor.run()

        outputs = []
        for output_tensor_handle in output_tensor_handles:
            output = output_tensor_handle.cpu()
            outputs.append(output)
        outputs_len = len(outputs)

        # 复制 fetch vars。如果 fetch 是 None，则从复制 output_tensor_handles 全部结果。
        # 否则，只从 output_tensor_handles 中复制指定的字段
        fetch_map = {}
        if fetch is None:
            for i, name in enumerate(output_names):
                fetch_map[name] = outputs[i]
                if len(output_tensor_handles[i].lod()) > 0:
                    fetch_map[name + ".lod"] = np.array(output_tensor_handles[i].lod()[0]).astype('int32')
        else:
            # 因为 save_inference_model 接口会增加网络中 scale op，fetch_var 会不同于 prototxt 的。
            # 因此兼容 v0.6.x 和之前的模型保存格式，并且兼容不匹配的结果
            fetch_match_num = 0
            for i, name in enumerate(fetch):
                output_index = output_name_to_index_dict.get(name)
                if output_index is None:
                    continue

                fetch_map[name] = outputs[output_index]
                fetch_match_num += 1
                if len(output_tensor_handles[output_index].lod()) > 0:
                    fetch_map[name + ".lod"] = np.array(output_tensor_handles[output_index].lod()[0]).astype('int32')

            # Compatible with v0.6.x and lower versions model saving formats.
            if fetch_match_num == 0:
                _LOGGER.debug("fetch match num is 0. Retrain the model please!")
                for i, name in enumerate(fetch):
                    if i >= outputs_len:
                        break
                    fetch_map[name] = outputs[i]
                    if len(output_tensor_handles[i].lod()) > 0:
                        fetch_map[name + ".lod"] = np.array(output_tensor_handles[i].lod()[0]).astype('int32')

        return fetch_map
