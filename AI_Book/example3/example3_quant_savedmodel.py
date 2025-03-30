import tensorflow as tf
import numpy as np
import time

import tensorflow_model_optimization as tfmot

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

t2_imgs=training_imgs[0:60000]
t2_labels=training_labels[0:60000]


print(f"Number of Evaluating Images: {len(test_imgs)}")

start_time = time.time()


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
  start_timens=time.time_ns()
  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_imgs):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i),end="")
      print("%s ns" % (time.time_ns() - start_timens))
      start_timens=time.time_ns()
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



print("Evaluating...")

interpreter = tf.lite.Interpreter(model_path='model_quat.tflite')
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print("--- %s seconds ---" % (time.time() - start_time))

