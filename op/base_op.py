import os

import torch
import cv2
import numpy as np

from op.operator import Op
from utils.logger import get_logger

_LOGGER = get_logger("Serving")


class ClsOp(Op):
    def init_op(self):
        self._post_func = torch.softmax

    def preprocess(self, input_dicts, data_id=0, log_id=0):
        """
        预处理，为 process 装配数据。用户可以重载本函数

        Args:
            input_dicts: 要被预处理的数据，格式为 {op_name: dict_data}
            data_id: 内部唯一 id，自增
            log_id: 全局唯一 id for RTT，默认 0

        Return:
            output_data: 给 process 的数据
            is_skip_process: 是否跳过 process，默认 False
            prod_errcode: 默认 None，否则发生业务错误。处理方式和异常一样
            prod_errinfo: 默认 ""
        """
        _LOGGER.debug("[ClsOp] preprocess() start")
        # multiple previous Op
        if len(input_dicts) != 1:
            _LOGGER.critical(self._log(
                "Failed to run preprocess: this Op has multiple previous inputs. Please override this func."))
            os._exit(-1)
        (_, input_dict), = input_dicts.items()
        raw_im = input_dict['img']
        data = np.frombuffer(raw_im, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_dict['img'] = rgb.transpose(2, 0, 1)[np.newaxis, :]
        return input_dict, False, None, ""

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
        _LOGGER.debug("[ClsOp] postprocess() start")
        fetch_dict = {}
        if isinstance(fetch_data, dict):
            tensor = fetch_data['prediction']
            preds = torch.softmax(tensor, dim=1).numpy()
            fetch_dict['prediction'] = preds.tobytes()
        return fetch_dict, None, ""


class SegOp(Op):
    def init_op(self):
        pass

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


class DetOp(Op):
    def init_op(self):
        pass

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
