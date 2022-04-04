import logging
import threading

import flask
from utils.logger import init_logger, get_logger, set_logger_level
from pipeline.pipeline_server import PipelineServer

_LOGGER = get_logger()


class Server(object):
    def __init__(self):
        self._server = PipelineServer()
        self._app_instance = None
        self.init_app()

    def start(self):
        _LOGGER.debug("[Server] init a flask in thread")
        p = threading.Thread(target=self.run_flask, args=())
        p.daemon = True
        p.start()
        _LOGGER.debug("[Server] run PipelineServer")
        self._server.run_server()

    def init_app(self):
        app_instance = flask.Flask(__name__)

        @app_instance.route("/index")
        def index():
            _LOGGER.debug("[Server] access app.index()")
            text = "text"
            return flask.Response(text, mimetype="text/plain")

        self._app_instance = app_instance

    def run_flask(self):
        _LOGGER.debug("[Server] app.run() start")
        self._app_instance.run(host="0.0.0.0", port=9898, threaded=True)
        _LOGGER.debug("[Server] app.run() end")


if __name__ == "__main__":
    init_logger(fpath="C:/Users/xf/Desktop/server2.log", mode='w')
    set_logger_level(level=logging.DEBUG)
    server = Server()
    server.start()
