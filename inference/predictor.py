from abc import ABC, abstractmethod

import timm
import torch

from utils.logger import get_logger

_LOGGER = get_logger("Serving")


class Predictor(ABC):
    def __init__(self, config):
        self._config = config
        self._predictor = None
        self._input_handle = None
        self._output_handle = None

    def _init_predictor(self):
        self._create_predictor()
        if self._config.use_gpu:
            self._predictor.cuda(self._config.gpuid)
        self._predictor.eval()

    @abstractmethod
    def _create_predictor(self):
        return NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @property
    def input_handle(self):
        return self._input_handle

    @input_handle.setter
    def input_handle(self, tensor):
        if tensor.device != self._predictor.device:
            self._input_handle = tensor.to(self._predictor.device)
        else:
            self._input_handle = tensor

    @property
    def output_handle(self):
        return self._output_handle


class Classifier(Predictor):
    def __init__(self, config):
        super(Classifier, self).__init__(config)

    def _create_predictor(self):
        self._predictor = timm.create_model(
            model_name=self._config.model_name,
            pretrained=False,
            checkpoint_path=self._config.checkpoint_path,
            num_classes=self._config.num_classes,
        )

    @torch.no_grad()
    def run(self):
        self._output_handle = self._predictor(self._input_handle)


def create_predictor(config):
    if config.model_type == 'rec':
        return Classifier(config)
    else:
        return NotImplementedError("no such model_type={}".format(config.model_type))
