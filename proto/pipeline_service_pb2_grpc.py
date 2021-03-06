# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import pipeline_service_pb2 as pipeline__service__pb2


class PipelineServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.inference = channel.unary_unary(
                '/PipelineService/inference',
                request_serializer=pipeline__service__pb2.Request.SerializeToString,
                response_deserializer=pipeline__service__pb2.Response.FromString,
                )


class PipelineServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def inference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PipelineServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'inference': grpc.unary_unary_rpc_method_handler(
                    servicer.inference,
                    request_deserializer=pipeline__service__pb2.Request.FromString,
                    response_serializer=pipeline__service__pb2.Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'PipelineService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PipelineService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PipelineService/inference',
            pipeline__service__pb2.Request.SerializeToString,
            pipeline__service__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
