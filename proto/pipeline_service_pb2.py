# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pipeline_service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16pipeline_service.proto\"2\n\x07Request\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05logid\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x03 \x03(\x0c\"X\n\x08Response\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05logid\x18\x02 \x01(\t\x12\x10\n\x08\x65rror_no\x18\x03 \x01(\x05\x12\x11\n\terror_msg\x18\x04 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x05 \x03(\x0c\x32\x35\n\x0fPipelineService\x12\"\n\tinference\x12\x08.Request\x1a\t.Response\"\x00\x62\x06proto3')



_REQUEST = DESCRIPTOR.message_types_by_name['Request']
_RESPONSE = DESCRIPTOR.message_types_by_name['Response']
Request = _reflection.GeneratedProtocolMessageType('Request', (_message.Message,), {
  'DESCRIPTOR' : _REQUEST,
  '__module__' : 'pipeline_service_pb2'
  # @@protoc_insertion_point(class_scope:Request)
  })
_sym_db.RegisterMessage(Request)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSE,
  '__module__' : 'pipeline_service_pb2'
  # @@protoc_insertion_point(class_scope:Response)
  })
_sym_db.RegisterMessage(Response)

_PIPELINESERVICE = DESCRIPTOR.services_by_name['PipelineService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REQUEST._serialized_start=26
  _REQUEST._serialized_end=76
  _RESPONSE._serialized_start=78
  _RESPONSE._serialized_end=166
  _PIPELINESERVICE._serialized_start=168
  _PIPELINESERVICE._serialized_end=221
# @@protoc_insertion_point(module_scope)
