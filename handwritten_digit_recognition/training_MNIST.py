import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# L1 regularization strength
l1_strength = 1e-5

# Define a quantization-friendly model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Quantization with representative dataset
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for i in range(1000):
        yield [x_train[i:i+1]]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# Convert and save
tflite_model = converter.convert()
with open("mnist_model_int8_relu.tflite", "wb") as f:
    f.write(tflite_model)

def convert_to_c_array(byte_data, var_name="model_data"):
    hex_array = ', '.join(f'0x{b:02x}' for b in byte_data)
    c_code = f"const unsigned char {var_name}[] = {{\n  {hex_array}\n}};\n"
    c_code += f"const int {var_name}_len = {len(byte_data)};\n"
    return c_code

with open("mnist_model_int8_relu.tflite", "rb") as f:
    model_bytes = f.read()

c_header = convert_to_c_array(model_bytes, "mnist_model_quant")

with open("mnist_model_quant.h", "w") as f:
    f.write(c_header)
