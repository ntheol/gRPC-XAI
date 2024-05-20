# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: xai_service.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11xai_service.proto\"+\n\x15InitializationRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t\"4\n\nTableModel\x12&\n\x07\x63ontent\x18\x01 \x01(\x0b\x32\x15.ExplanationsResponse\"\xe5\x01\n\x05Model\x12\x15\n\rfeature_names\x18\x01 \x03(\t\x12 \n\x05plots\x18\x02 \x03(\x0b\x32\x11.Model.PlotsEntry\x12\"\n\x06tables\x18\x03 \x03(\x0b\x32\x12.Model.TablesEntry\x1a\x43\n\nPlotsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.ExplanationsResponse:\x02\x38\x01\x1a:\n\x0bTablesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1a\n\x05value\x18\x02 \x01(\x0b\x32\x0b.TableModel:\x02\x38\x01\"L\n\x16InitializationResponse\x12\x15\n\x05model\x18\x01 \x01(\x0b\x32\x06.Model\x12\x1b\n\x08pipeline\x18\x02 \x01(\x0b\x32\t.Pipeline\"\xf5\x01\n\x08Pipeline\x12\x1c\n\x14hyperparameter_names\x18\x01 \x03(\t\x12#\n\x05plots\x18\x02 \x03(\x0b\x32\x14.Pipeline.PlotsEntry\x12%\n\x06tables\x18\x03 \x03(\x0b\x32\x15.Pipeline.TablesEntry\x1a\x43\n\nPlotsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.ExplanationsResponse:\x02\x38\x01\x1a:\n\x0bTablesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1a\n\x05value\x18\x02 \x01(\x0b\x32\x0b.TableModel:\x02\x38\x01\"\xdf\x01\n\x13\x45xplanationsRequest\x12\x18\n\x10\x65xplanation_type\x18\x01 \x01(\t\x12\x1a\n\x12\x65xplanation_method\x18\x02 \x01(\t\x12\r\n\x05model\x18\x03 \x01(\t\x12\x10\n\x08\x66\x65\x61ture1\x18\x04 \x01(\t\x12\x10\n\x08\x66\x65\x61ture2\x18\x05 \x01(\t\x12\x17\n\x0fnum_influential\x18\x06 \x01(\x05\x12\x15\n\rproxy_dataset\x18\x07 \x01(\x0c\x12\r\n\x05query\x18\x08 \x01(\x0c\x12\x10\n\x08\x66\x65\x61tures\x18\t \x01(\t\x12\x0e\n\x06target\x18\n \x01(\t\".\n\x08\x46\x65\x61tures\x12\x10\n\x08\x66\x65\x61ture1\x18\x01 \x01(\t\x12\x10\n\x08\x66\x65\x61ture2\x18\x02 \x01(\t\"A\n\x04\x41xis\x12\x11\n\taxis_name\x18\x01 \x01(\t\x12\x13\n\x0b\x61xis_values\x18\x02 \x03(\t\x12\x11\n\taxis_type\x18\x03 \x01(\t\"\x86\x02\n\x14\x45xplanationsResponse\x12\x1b\n\x13\x65xplainability_type\x18\x01 \x01(\t\x12\x1a\n\x12\x65xplanation_method\x18\x02 \x01(\t\x12\x1c\n\x14\x65xplainability_model\x18\x03 \x01(\t\x12\x11\n\tplot_name\x18\x04 \x01(\t\x12\x12\n\nplot_descr\x18\x05 \x01(\t\x12\x11\n\tplot_type\x18\x06 \x01(\t\x12\x1b\n\x08\x66\x65\x61tures\x18\x07 \x01(\x0b\x32\t.Features\x12\x14\n\x05xAxis\x18\x08 \x01(\x0b\x32\x05.Axis\x12\x14\n\x05yAxis\x18\t \x01(\x0b\x32\x05.Axis\x12\x14\n\x05zAxis\x18\n \x01(\x0b\x32\x05.Axis2\x92\x01\n\x0c\x45xplanations\x12?\n\x0eGetExplanation\x12\x14.ExplanationsRequest\x1a\x15.ExplanationsResponse(\x01\x12\x41\n\x0eInitialization\x12\x16.InitializationRequest\x1a\x17.InitializationResponseB\x13\n\x11gr.grpc.generatedb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'xai_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\021gr.grpc.generated'
  _globals['_MODEL_PLOTSENTRY']._options = None
  _globals['_MODEL_PLOTSENTRY']._serialized_options = b'8\001'
  _globals['_MODEL_TABLESENTRY']._options = None
  _globals['_MODEL_TABLESENTRY']._serialized_options = b'8\001'
  _globals['_PIPELINE_PLOTSENTRY']._options = None
  _globals['_PIPELINE_PLOTSENTRY']._serialized_options = b'8\001'
  _globals['_PIPELINE_TABLESENTRY']._options = None
  _globals['_PIPELINE_TABLESENTRY']._serialized_options = b'8\001'
  _globals['_INITIALIZATIONREQUEST']._serialized_start=21
  _globals['_INITIALIZATIONREQUEST']._serialized_end=64
  _globals['_TABLEMODEL']._serialized_start=66
  _globals['_TABLEMODEL']._serialized_end=118
  _globals['_MODEL']._serialized_start=121
  _globals['_MODEL']._serialized_end=350
  _globals['_MODEL_PLOTSENTRY']._serialized_start=223
  _globals['_MODEL_PLOTSENTRY']._serialized_end=290
  _globals['_MODEL_TABLESENTRY']._serialized_start=292
  _globals['_MODEL_TABLESENTRY']._serialized_end=350
  _globals['_INITIALIZATIONRESPONSE']._serialized_start=352
  _globals['_INITIALIZATIONRESPONSE']._serialized_end=428
  _globals['_PIPELINE']._serialized_start=431
  _globals['_PIPELINE']._serialized_end=676
  _globals['_PIPELINE_PLOTSENTRY']._serialized_start=223
  _globals['_PIPELINE_PLOTSENTRY']._serialized_end=290
  _globals['_PIPELINE_TABLESENTRY']._serialized_start=292
  _globals['_PIPELINE_TABLESENTRY']._serialized_end=350
  _globals['_EXPLANATIONSREQUEST']._serialized_start=679
  _globals['_EXPLANATIONSREQUEST']._serialized_end=902
  _globals['_FEATURES']._serialized_start=904
  _globals['_FEATURES']._serialized_end=950
  _globals['_AXIS']._serialized_start=952
  _globals['_AXIS']._serialized_end=1017
  _globals['_EXPLANATIONSRESPONSE']._serialized_start=1020
  _globals['_EXPLANATIONSRESPONSE']._serialized_end=1282
  _globals['_EXPLANATIONS']._serialized_start=1285
  _globals['_EXPLANATIONS']._serialized_end=1431
# @@protoc_insertion_point(module_scope)
