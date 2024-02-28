import numpy as np
import sys
sys.path.append("./src/")
import tensorflow as tf
from download import preprocessor

X_train, y_train, X_test, y_test = preprocessor.tiny_sample()

QUANTIZED_MODEL_TFLITE = f'saved_models/tinycnn/model.tflite'
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_TFLITE)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
correct = 0
for index in range(10):
    input_data = np.array(X_test[index:index+1], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(np.argmax(output_data, axis=1), output_data)
    print(y_test[index])
    if np.argmax(output_data, axis=1) == y_test[index]:
        correct += 1 
print(correct)