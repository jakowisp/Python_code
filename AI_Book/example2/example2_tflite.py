import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)



data=tf.keras.datasets.fashion_mnist
(training_imgs,training_labels),(test_imgs,test_labels) = data.load_data()

print("Normalizing data")
training_imgs = training_imgs /255.0
test_imgs = test_imgs/255.0

t2_imgs=training_imgs[0:60000]
t2_labels=training_labels[0:60000]


print(f"Number of Training Images:   {len(training_imgs)}")
print(f"Number of Evaluating Images: {len(test_imgs)}")

print("Defining Nueral Network")
l0 = Flatten(input_shape=(28,28))
l1 = Dense(128,activation=tf.nn.relu)
l2 = Dense(10,activation=tf.nn.softmax)
model = Sequential([l0,l1,l2])
print("Compiling Model")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

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

print("Fitting data to labels")
#model.fit(training_imgs, training_labels, epochs=5, use_multiprocessing=True, callbacks=[Callback()], verbose=0)
#model.fit(training_imgs, training_labels, epochs=50)
model.fit(t2_imgs, t2_labels, epochs=5, batch_size=128)

tf.saved_model.save(model, "saved_model_keras_dir")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Evaluating...")
model.evaluate(test_imgs,test_labels)
print("--- %s seconds ---" % (time.time() - start_time))

