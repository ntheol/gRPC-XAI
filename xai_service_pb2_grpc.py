# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import xai_service_pb2 as xai__service__pb2


class ExplanationsStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetExplanation = channel.unary_unary(
                '/Explanations/GetExplanation',
                request_serializer=xai__service__pb2.ExplanationsRequest.SerializeToString,
                response_deserializer=xai__service__pb2.ExplanationsResponse.FromString,
                )


class ExplanationsServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetExplanation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExplanationsServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetExplanation': grpc.unary_unary_rpc_method_handler(
                    servicer.GetExplanation,
                    request_deserializer=xai__service__pb2.ExplanationsRequest.FromString,
                    response_serializer=xai__service__pb2.ExplanationsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Explanations', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Explanations(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetExplanation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Explanations/GetExplanation',
            xai__service__pb2.ExplanationsRequest.SerializeToString,
            xai__service__pb2.ExplanationsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class InfluencesStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ComputeInfluences = channel.stream_unary(
                '/Influences/ComputeInfluences',
                request_serializer=xai__service__pb2.InfluenceRequest.SerializeToString,
                response_deserializer=xai__service__pb2.InfluenceResponse.FromString,
                )


class InfluencesServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ComputeInfluences(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InfluencesServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ComputeInfluences': grpc.stream_unary_rpc_method_handler(
                    servicer.ComputeInfluences,
                    request_deserializer=xai__service__pb2.InfluenceRequest.FromString,
                    response_serializer=xai__service__pb2.InfluenceResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Influences', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Influences(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ComputeInfluences(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/Influences/ComputeInfluences',
            xai__service__pb2.InfluenceRequest.SerializeToString,
            xai__service__pb2.InfluenceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
