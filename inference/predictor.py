from abc import ABC, abstractmethod

import timm
import torch
from torchvision import transforms

from utils.logger import get_logger

_LOGGER = get_logger("Serving")


class Predictor(ABC):
    def __init__(self, config):
        self._config = config
        self._transforms = None
        self._predictor = None
        self._input_handle = {name: None for name in self._config.input_names}
        self._output_handle = {name: None for name in self._config.output_names}
        self._device = torch.device(self._config.device)
        self._init_predictor()

    def _init_predictor(self):
        self._create_predictor()
        self._predictor.to(self._device)
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

    def set_input_handle(self, name, ndarray):
        tensor = torch.from_numpy(ndarray).contiguous()
        if self._transforms is not None:
            tensor = self._transforms(tensor)
        if tensor.device != self._device:
            self._input_handle[name] = tensor.to(self._device)
        else:
            self._input_handle[name] = tensor

    @property
    def output_handle(self):
        return self._output_handle

    @property
    def device(self):
        return self._device

    def get_input_names(self):
        return self._config.input_names

    def get_input_handle(self, name):
        return self._input_handle[name]

    def get_output_names(self):
        return self._config.output_names

    def get_output_handle(self, name):
        return self._output_handle[name]


class Classifier(Predictor):
    def __init__(self, config):
        super(Classifier, self).__init__(config)
        self._transforms = transforms.Compose([
            transforms.Normalize(mean=self._config.mean, std=self._config.std),
        ])

    def _create_predictor(self):
        self._predictor = timm.create_model(
            model_name=self._config.model_name,
            pretrained=False,
            checkpoint_path=self._config.checkpoint_path,
            num_classes=self._config.num_classes,
        )

    @torch.no_grad()
    def run(self):
        self._output_handle[self._config.output_names[0]] = \
            self._predictor(self._input_handle[self._config.input_names[0]])


def create_predictor(config):
    if config.model_type == 'rec':
        classifier = Classifier(config)
        _LOGGER.info("create a Classifier: {} on {}".format(classifier, classifier.device))
        return classifier
    else:
        return NotImplementedError("no such model_type={}".format(config.model_type))
