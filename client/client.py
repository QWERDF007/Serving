import logging
import grpc
from proto import pipeline_service_pb2, pipeline_service_pb2_grpc
from utils.logger import init_logger, get_logger, set_logger_level

_LOGGER = get_logger("Client")


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

    def start(self):
        while True:
            request = pipeline_service_pb2.Request()
            response = self._stub.inference(request)
            _LOGGER.debug(f"[Client] get response from server response={response}")


if __name__ == "__main__":
    _LOGGER.debug("Client entry")
    init_logger("Client", fpath="C:/Users/wt/Desktop/client.log", mode='w')
    set_logger_level("Client", level=logging.DEBUG)
    client = Client()
    client.start()
