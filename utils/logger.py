import logging


def init_logger(name='Serving', fpath=None, mode='a'):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler()]
    if fpath is not None:
        handlers.append(logging.FileHandler(fpath, mode=mode))
    fomatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(thread)d] [%(levelname)1.1s] %(message)s")
    for handler in handlers:
        handler.setFormatter(fomatter)
        logger.addHandler(handler)


def get_logger(name="Serving"):
    return logging.getLogger(name)


def set_logger_fmt(name="Serving", fmt=""):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt)
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def set_logger_level(name="Serving", level: [int, str] = logging.INFO):
    logger = logging.getLogger(name)
    # "[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] [%(process)d] [%(thread)d] [%(levelname)1.1s] %(message)s"
    if level == logging.DEBUG:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(process)6d] [%(thread)6d] [%(filename)25s:%(lineno)5d] [%(levelname)1.1s] %(message)s")
        for handler in logger.handlers:
            # handler.setLevel(level)
            handler.setFormatter(formatter)
    elif logger.level == logging.DEBUG and level != logging.DEBUG:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(thread)d] [%(levelname)1.1s] %(message)s")
        for handler in logger.handlers:
            # handler.setLevel(level)
            handler.setFormatter(formatter)
    else:
        pass
    logger.setLevel(level)
