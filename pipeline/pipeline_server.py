import grpc
from concurrent import futures
from utils.logger import get_logger

from proto import pipeline_service_pb2, pipeline_service_pb2_grpc

_LOGGER = get_logger()


class PipelineServicer(pipeline_service_pb2_grpc.PipelineServiceServicer):
    def __init__(self, name: str, response_op, dag_conf, worker_idx=-1):
        super(PipelineServicer, self).__init__()

    def inference(self, request, context):
        _LOGGER.debug(f"[PipelineServicer] inference get peer: {context.peer()}")
        response = pipeline_service_pb2.Response()
        return response


class PipelineServer(object):
    def __init__(self):
        self._name = None
        self._worker_num = None
        self._rpc_port = 9393
        self._conf = None
        self._response_op = None

    def prepare_server(self, config):
        pass

    def run_server(self):
        server = grpc.server(
            thread_pool=futures.ThreadPoolExecutor(max_workers=self._worker_num),
            options=[('grpc.max_send_message_length', 512 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
        pipeline_service_pb2_grpc.add_PipelineServiceServicer_to_server(
            PipelineServicer(self._name, self._response_op, self._conf), server)
        server.add_insecure_port('[::]:{}'.format(self._rpc_port))
        _LOGGER.debug("[PipelineServer] grpc.server.start()")
        server.start()
        # self._run_grpc_gateway(grpc_port=self._rpc_port, http_port=self._http_port)  # start grpc_gateway
        _LOGGER.debug(f"[PipelineServer] grpc.server.wait_for_termination() on port:{self._rpc_port}")
        server.wait_for_termination()
