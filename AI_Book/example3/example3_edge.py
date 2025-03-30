import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow_model_optimization.python.core.keras.compat import keras

quantize_model = tfmot.quantization.keras.quantize_model

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)



data=tf.keras.datasets.fashion_mnist
(training_imgs,training_labels),(test_imgs,test_labels) = data.load_data()

print("Normalizing data")
training_imgs = training_imgs.reshape(60000,28,28,1)
training_imgs = training_imgs /255.0
test_imgs = test_imgs.reshape(10000,28,28,1)
test_imgs = test_imgs/255.0

print(f"Number of Training Images:   {len(training_imgs)}")
print(f"Number of Evaluating Images: {len(test_imgs)}")

print("Defining Nueral Network")
l0 = Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))
l1 = MaxPooling2D(2,2)
l2 = Conv2D(64, (3,3), activation='relu')
l3 = MaxPooling2D(2,2)
l4 = Flatten()
l5 = Dense(128,activation=tf.nn.relu)
l6 = Dense(10,activation=tf.nn.softmax)
model = Sequential([l0,l1,l2,l3,l4,l5,l6])

start_time = time.time()
print("Compiling  Quatization Model")

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

q_aware_model.summary()

class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 1
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.counter == self.SHOW_NUMBER or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
            if self.epoch > 1:
                self.counter = 0
        self.counter += 1

def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_imgs):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy


print("Fitting data to labels - quatizations")
train_images_subset = training_imgs[0:60000] # out of 60000
train_labels_subset = training_labels[0:60000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=128, epochs=5, validation_split=0.1)

def representative_data_gen():

    mnist_train, _ = tf.keras.datasets.fashion_mnist.load_data()
    images = tf.cast(mnist_train[0].reshape(60000,28,28,1), tf.float32) / 255.0
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

convert_model(q_aware_model,representative_data_gen,'qaware_ex3.tflite')

print("Evaluating...")
_, q_aware_model_accuracy = q_aware_model.evaluate( test_imgs, test_labels, verbose=0)

interpreter = tf.lite.Interpreter(model_path='qaware_ex3.tflite')
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)
print("--- %s seconds ---" % (time.time() - start_time))

