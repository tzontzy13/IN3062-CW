import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

import matplotlib.pyplot as plt

import pathlib

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
import tensorflow.keras.losses as losses

from sklearn.model_selection import train_test_split

from skimage.color import rgb2gray

# load array
X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')

# y_set = tensorflow.keras.utils.to_categorical(y_set)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_set, y_set, test_size=0.20, random_state=42)


model = Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.Dense(1000, activation='relu'),
    layers.Dense(700, activation='relu'),
    layers.Dense(120, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# model.add(Flatten(input_shape=(100, 100)))
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(10, activation='softmax'))

custom = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=custom,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.summary()
BATCH_SIZE = 35
history = model.fit(rgb2gray(X_train), y_train,
                    # validation_split=0.2,
                    epochs=100, batch_size=35, verbose=2)

model.evaluate(rgb2gray(X_test),  y_test, verbose=2)
