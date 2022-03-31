import logging

from utils.logger import init_logger, get_logger, set_logger_level
from pipeline.pipeline_server import PipelineServer

_LOGGER = get_logger()


class Server(object):
    def __init__(self):
        self._server = PipelineServer()

    def start(self):
        self._server.run_server()


if __name__ == "__main__":
    init_logger(fpath="C:/Users/wt/Desktop/server.log", mode='w')
    set_logger_level(level=logging.DEBUG)
    server = Server()
    server.start()
