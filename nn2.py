from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
import tensorflow.keras.losses as losses

# X_set = np.load('data_set_images.npy')
# y_set = np.load('data_set_targets.npy')

# y_set = tf.keras.utils.to_categorical(y_set)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_set, y_set, test_size=0.20, random_state=42)

X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')

# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

# y_train = np.reshape(y_train, (y_train.shape[0], 1))
# y_test = np.reshape(y_test, (y_test.shape[0], 1))

X_train = rgb2gray(X_train)
X_test = rgb2gray(X_test)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(
    number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(
    number_of_test, X_test.shape[1]*X_test.shape[2])

# model = Sequential([
#     layers.Flatten(input_shape=(100, 100)),
#     # layers.Dense(256, activation='relu'),
#     # layers.Dropout(0.2),
#     layers.Dense(512+256+16, activation='relu'),
#     # layers.Dense(256, activation='relu'),
#     layers.Dense(128+256+16, activation='relu'),
#     # layers.Dropout(0.2),
#     layers.Dense(64+32+16, activation='relu'),
#     layers.Dense(10) # , activation='softmax'
#     ])

model = Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512+128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') #, activation='softmax'
    ])

custom = tf.keras.optimizers.Adam(learning_rate=0.0000051)
custom2 = tf.keras.optimizers.SGD(0.02, 0.5)

model.compile(optimizer=custom,
            #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.summary()

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=100, batch_size=40,
                    shuffle=True,
                    verbose=2)

model.evaluate(X_test,  y_test) # , verbose=2