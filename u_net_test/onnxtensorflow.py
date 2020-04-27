import onnx
from onnx_tf.backend import prepare

# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unet_tunedlatest.onnx' )

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('/home/psridharan/Unet-Berkan/u_net_test/saved_model/Unet_tuned.pb')