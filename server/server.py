import argparse
import logging
import os.path
import threading

import flask
from utils.logger import init_logger, get_logger, set_logger_level
from pipeline.pipeline_server import PipelineServer

_LOGGER = get_logger()


def server_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--conf', type=str, metavar='PATH', help='path to server yml conf')
    parser.add_argument('--logs-dir', type=str, metavar='DIR', default='./', help='directory to save logs')
    parser.add_argument('--log-mode', type=str, metavar='MODE', default='w', help='mode of log file')
    parser.add_argument('--log-level', type=int, metavar='LEVEL', default=logging.DEBUG,
                        choices=[0, 10, 20, 30, 40, 50],
                        help='the numeric values of logging levels. 10 means DEBUG, 20 means INFO')
    return parser.parse_args()


class Server(object):
    def __init__(self, yml_file):
        self._server = PipelineServer()
        self._app_instance = None
        # self.init_app()
        self._prepare_server(yml_file)

    def init_app(self):
        app_instance = flask.Flask(__name__)

        @app_instance.route("/index")
        def index():
            _LOGGER.debug("[Server] access app.index()")
            text = "text"
            return flask.Response(text, mimetype="text/plain")

        self._app_instance = app_instance

    def _run_flask(self):
        self._app_instance.run(host="0.0.0.0", port=9898, threaded=True)

    def _prepare_server(self, yml_file=None):
        _LOGGER.debug("[Server] _prepare_server() start")
        # TODO load yml conf
        self._load_conf()
        # TODO init ResponseOp
        response_op = self._init_ops()
        self._server.set_response_op(response_op)
        self._server.prepare_server(yml_file)

    def _load_conf(self):
        pass

    def _init_ops(self):
        pass

    def run_server(self):
        self._server.run_server()


if __name__ == "__main__":
    args = server_args()
    init_logger(fpath=os.path.join(args.logs_dir, 'server.log'), mode=args.log_mode)
    set_logger_level(level=args.log_level)
    server = Server(args.conf)
    server.run_server()
