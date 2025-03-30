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
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
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
train_images_subset = training_imgs[0:10000] # out of 60000
train_labels_subset = training_labels[0:10000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=128, epochs=5, validation_split=0.1)

converter2 = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
quat_model = converter2.convert()
with open('model_quat.tflite', 'wb') as f:
    f.write(quat_model)

print("Evaluating...")
_, q_aware_model_accuracy = q_aware_model.evaluate( test_imgs, test_labels, verbose=0)

interpreter = tf.lite.Interpreter(model_content=quat_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)
print("--- %s seconds ---" % (time.time() - start_time))

