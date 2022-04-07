import io
import yaml
from utils.logger import get_logger

_LOGGER = get_logger("Serving")


class Config(object):
    def __init__(self, yml_file):
        with io.open(yml_file, encoding='utf-8') as f:
            model_conf = yaml.load(f.read(), yaml.FullLoader)

        Config.check_model_conf(model_conf)
        self._input_names = model_conf['input_names']
        self._output_names = model_conf['output_names']
        self._model_name = model_conf['model_name']
        self._model_type = model_conf['model_type']
        self._checkpoint_path = model_conf['checkpoint_path']
        self._num_classes = model_conf['num_classes']
        self._mean = model_conf['mean']
        self._std = model_conf['std']
        self._use_gpu = False
        self._device = 'cpu'

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def device(self):
        return self._device

    @property
    def use_gpu(self):
        return self._use_gpu

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    def disable_gpu(self):
        self._use_gpu = True
        self._device = 'cpu'

    def enable_use_gpu(self, gpu_id):
        self._use_gpu = True
        self._device = 'cuda:{}'.format(gpu_id)

    @staticmethod
    def check_model_conf(conf):
        default_conf = {
        }

        conf_type = {
            "model_name": str,
            "model_type": str,
            "checkpoint_path": str,
            "num_classes": int,
        }

        conf_qualification = {
            "model_type": [("in", ["rec", "seg", "det"])],
            "num_classes": [(">=", 1), ("<=", 65536)],
        }

        Config.check_conf(conf, default_conf, conf_type, conf_qualification)

    @staticmethod
    def check_conf(conf, default_conf, conf_type, conf_qualification):
        Config.fill_with_default_conf(conf, default_conf)
        Config.check_conf_type(conf, conf_type)
        Config.check_conf_qualification(conf, conf_qualification)

    @staticmethod
    def fill_with_default_conf(conf, default_conf):
        for key, val in default_conf.items():
            if conf.get(key) is None:
                _LOGGER.warning("[CONF] {} not set, use default: {}".format(key, val))
                conf[key] = val

    @staticmethod
    def check_conf_type(conf, conf_type):
        for key, val in conf_type.items():
            if key not in conf:
                continue
            if not isinstance(conf[key], val):
                raise SystemExit("[CONF] {} must be {} type, but get {}.".format(key, val, type(conf[key])))

    @staticmethod
    def check_conf_qualification(conf, conf_qualification):
        for key, qualification in conf_qualification.items():
            if key not in conf:
                continue
            if not isinstance(qualification, list):
                qualification = [qualification]
            if not Config.qualification_check(conf[key], qualification):
                raise SystemExit(
                    "[CONF] {} must be {}, but get {}.".format(
                        key, ", ".join(["{} {}".format(q[0], q[1]) for q in qualification]), conf[key]))

    @staticmethod
    def qualification_check(value, qualifications):
        if not isinstance(qualifications, list):
            qualifications = [qualifications]
        ok = True
        for q in qualifications:
            operator, limit = q
            if operator == "<":
                ok = value < limit
            elif operator == "==":
                ok = value == limit
            elif operator == ">":
                ok = value > limit
            elif operator == "<=":
                ok = value <= limit
            elif operator == ">=":
                ok = value >= limit
            elif operator == "in":
                ok = value in limit
            else:
                raise SystemExit("unknown operator: {}".format(operator))
            if not ok:
                break
        return ok
