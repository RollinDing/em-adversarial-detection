import sys
sys.path.append("./src/")
from matplotlib.pyplot import quiverkey

from download import preprocessor
from models.classify import classifyTinyCNN
from sklearn.metrics import confusion_matrix
import numpy as np 
import tensorflow as tf
from utils.save import *
from sklearn.preprocessing import OneHotEncoder

name = "tinycnn-2layers"
X_train, y_train, X_test, y_test = preprocessor.tiny_sample()
# X_train = X_train.reshape([X_train.shape[0], -1])
# X_test  = X_test.reshape([X_test.shape[0], -1])
# input_shape = X_train.shape[1:]


# save_array(X_train, "data/downsampled/X_train.npy")
# save_array(y_train, "data/downsampled/y_train.npy")
# save_array(X_test, "data/downsampled/X_test.npy")
# save_array(y_test, "data/downsampled/y_test.npy")

model = classifyTinyCNN(X_train, y_train, X_test, y_test, output_directory=f"saved_models/{name}/")
MODEL_TF = f"saved_models/{name}/model"
model.save(MODEL_TF)
model.summary()


MODEL_NO_QUANT_TFLITE =f"saved_models/{name}/model_no_quant.tflite"
MODEL_TFLITE = f"saved_models/{name}/model.tflite"
MODEL_TFLITE_MICRO = f"saved_models/{name}/model.cc"

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
model_no_quant_tflite = converter.convert()

# Save the model to disk
open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

# Convert the model to the TensorFlow Lite format with quantization
def representative_dataset():
  for i in range(500):
    yield([np.expand_dims(X_train[i], axis=0)])
# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(MODEL_TFLITE, "wb").write(model_tflite)

def predict_tflite(tflite_model, x_test):
  # Prepare the test data
  x_test_ = x_test.copy()
  # x_test_ = x_test_.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
  x_test_ = x_test_.astype(np.float32)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # If required, quantize the input layer (from float to integer)
  input_scale, input_zero_point = input_details["quantization"]
  if (input_scale, input_zero_point) != (0.0, 0):
    x_test_ = x_test_ / input_scale + input_zero_point
    x_test_ = x_test_.astype(input_details["dtype"])

  print(x_test_)
  save_array(x_test_, "data/downsampled/X_quantized.npy")

  # Invoke the interpreter
  y_pred = np.empty([x_test_.shape[0], y_test.shape[1]], dtype=output_details["dtype"])
  for i in range(len(x_test_)):
    interpreter.set_tensor(input_details["index"], [x_test_[i]])
    interpreter.invoke()
    y_pred[i] = interpreter.get_tensor(output_details["index"])[0]
  
  # If required, dequantized the output layer (from integer to float)
  output_scale, output_zero_point = output_details["quantization"]
  if (output_scale, output_zero_point) != (0.0, 0):
    y_pred = y_pred.astype(np.float32)
    y_pred = (y_pred - output_zero_point) * output_scale

  return y_pred

def evaluate_tflite(tflite_model, x_test, y_true):
  global model
  y_pred = predict_tflite(tflite_model, x_test)
  loss_function = tf.keras.losses.get(model.loss)
  loss = loss_function(y_true, y_pred).numpy()
  return loss

enc = OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

y_test_pred_tf = model.predict(X_test)
y_test_pred_no_quant_tflite = predict_tflite(model_no_quant_tflite, X_test)
y_test_pred_tflite = predict_tflite(model_tflite, X_test)
print(y_test_pred_no_quant_tflite, y_test_pred_tflite)

import os

os.system(f"xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}")
REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
os.system(f"sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}")