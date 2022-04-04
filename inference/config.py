import io
import yaml
from utils.logger import get_logger

_LOGGER = get_logger("Serving")


class Config(object):
    def __init__(self, yml_file):
        with io.open(yml_file, encoding='utf-8') as f:
            self._conf = yaml.load(f.read(), yaml.FullLoader)

        Config.check_model_conf(self._conf)

    @staticmethod
    def check_model_conf(conf):
        default_conf = {
            "use_gpu": False,
            "gpu_id": -1,
        }

        conf_type = {
            "use_gpu": bool,
            "gpu_id": int,
            "checkpoint_path": str,
            "num_classes": int,
        }

        conf_qualification = {
            "gpu_id": [(">=", 0), ("<=", 65536)],
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
