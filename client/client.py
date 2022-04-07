import argparse
import logging
import os.path

import grpc
import cv2
import numpy as np
from proto import pipeline_service_pb2, pipeline_service_pb2_grpc
from utils.logger import init_logger, get_logger, set_logger_level
from utils.network_util import get_local_ip

_LOGGER = get_logger("Client")


def client_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ip', type=str, help='grpc server ip')
    parser.add_argument('--port', type=str, help='grpc server port')
    parser.add_argument('--logs-dir', type=str, default='./logs', metavar='DIR', help='directory to save logs')
    parser.add_argument('--log-mode', type=str, default='w', metavar='MODE', help='mode of log file')
    parser.add_argument('--log-level', type=int, metavar='LEVEL', default=logging.DEBUG,
                        choices=[0, 10, 20, 30, 40, 50],
                        help='the numeric values of logging levels. 10 means DEBUG, 20 means INFO')
    return parser.parse_args()


class Client(object):
    def __init__(self, ip="127.0.0.1", port="9393"):
        self._ip = ip
        self._port = port
        endpoints = [self._ip + ":" + self._port]
        self._endpoints = "ipv4:{}".format(",".join(endpoints))
        self._channel = grpc.insecure_channel(
            target=self._endpoints,
            options=[('grpc.max_send_message_length', 512 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
        self._stub = pipeline_service_pb2_grpc.PipelineServiceStub(self._channel)
        self._clentip = get_local_ip()

    def _preprocess(self):
        pass

    def _postprocess(self, response):
        for v in response.value:
            prediction = np.frombuffer(v, dtype=np.float32).reshape(1, -1)
            classification = prediction.argmax(axis=1)
            _LOGGER.info("prediction: {}, classification: {}".format(prediction, classification))

    def start(self):
        for i in range(10):
            request = pipeline_service_pb2.Request()
            request.logid = "1"
            request.clientip = self._clentip
            request.key.append('img')
            img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
            ret, img_encoded = cv2.imencode('.png', img)
            request.value.append(img_encoded.tobytes())
            response = self._stub.inference(request)
            self._postprocess(response)
            _LOGGER.debug(f"[Client] get response from server")


if __name__ == "__main__":
    args = client_args()
    init_logger("Client", fpath=os.path.join(args.logs_dir, 'client.log'), mode=args.log_mode)
    set_logger_level("Client", level=args.log_level)
    client = Client(args.ip, args.port)
    client.start()
