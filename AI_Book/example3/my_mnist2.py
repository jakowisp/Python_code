import tensorflow as tf
import os
import numpy as np
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
# Load MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.

l0 = tf.keras.layers.Flatten(input_shape=(28,28))
l1 = tf.keras.layers.Dense(128,activation=tf.nn.relu)
l2 = tf.keras.layers.Dense(10,activation=tf.nn.softmax)
model = tf.keras.Sequential([l0,l1,l2])

print("Compiling  Quatization Model")

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(train_images, train_labels,
                  batch_size=128, epochs=5, validation_split=0.1, validation_data=(test_images,test_labels))

def representative_data_gen():
    mnist_train, _ = tf.keras.datasets.fashion_mnist.load_data()
    images = tf.cast(mnist_train[0], tf.float32) / 255.0
    mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
    for input_value in mnist_ds.take(100):
    # Model has only one input so each data point has one element.
        yield [input_value]

def convert_model(model,data,filename):
    print(f"{filename}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # And this sets the representative dataset so we can quantize the activations
    converter.representative_dataset = representative_data_gen
    #converter.experimental_new_converter = False
    tflite_model = converter.convert()

    with open(filename, 'wb') as f:
        f.write(tflite_model)

convert_model(q_aware_model,representative_data_gen,'qaware_mnist_quant.tflite')

