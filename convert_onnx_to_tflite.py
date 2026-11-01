import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

print("Loading ONNX model...")
onnx_model = onnx.load("rawnet2.onnx")

print("Converting ONNX to TF SavedModel...")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model_onnx_tf")

print("Converting TF SavedModel to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_onnx_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

print("Saving TFLite model...")
with open("rawnet2_float16.tflite", "wb") as f:
    f.write(tflite_model)
print("Done! Saved to rawnet2_float16.tflite")
