import tensorflow as tf
import numpy as np
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

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
print("Compiling Model")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

start_time = time.time()


print("Fitting data to labels")
model.fit(test_imgs, test_labels, epochs=5, batch_size=128)
print("Evaluating...")
model.evaluate(test_imgs,test_labels)
print("--- %s seconds ---" % (time.time() - start_time))

