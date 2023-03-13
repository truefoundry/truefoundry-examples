import argparse
from google.protobuf.json_format import MessageToDict
import grpc
from tensorflow_serving.apis import predict_pb2, get_model_metadata_pb2, prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host",
    type=str,
    required=True,
    help="Host of the deployed tf serving service",
)
args = parser.parse_args()

GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle

channel = grpc.secure_channel(
        args.host,
        credentials=grpc.ssl_channel_credentials(),
)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# GetModelMetadata
grpc_request = get_model_metadata_pb2.GetModelMetadataRequest()
grpc_request.model_spec.name = "mobilenet-v3-small"
grpc_request.model_spec.signature_name = "serving_default"
grpc_request.metadata_field.append("signature_def")
output = stub.GetModelMetadata(grpc_request)
print(MessageToDict(output))
print("=" * 100)

# Predict example
image = tf.io.read_file("german-shepherd.jpeg")
image = tf.io.decode_jpeg(image)
image = tf.image.convert_image_dtype(image, dtype=tf.int8)
image = tf.expand_dims(image, 0)
grpc_request = predict_pb2.PredictRequest()
grpc_request.model_spec.name = 'mobilenet-v3-small'
grpc_request.model_spec.signature_name = 'serving_default'
grpc_request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(image, dtype=tf.float32))
predictions = stub.Predict(grpc_request, 10.0)
outputs_tensor_proto = predictions.outputs['logits']
shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
outputs = tf.nn.softmax(outputs, axis=1)
top = tf.argmax(outputs, axis=1).numpy()[0]
print("Predicted Class:", top, outputs[0][top])
