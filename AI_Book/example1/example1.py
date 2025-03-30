import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

print("Here is what I started with: {}".format(l0.get_weights()))
xs = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],dtype=float)
ys = np.asarray([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0],dtype=float)
start_time = time.time()

class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 100
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

#model.fit(xs, ys, epochs=5000, use_multiprocessing=True, callbacks=[Callback()], verbose=0)
model.fit(xs, ys, epochs=50 )

print(model.predict([10.0]))
print("Here is what I learned: {}".format(l0.get_weights()))
print("--- %s seconds ---" % (time.time() - start_time))
