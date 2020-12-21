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
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
import tensorflow.keras.losses as losses
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

# load array
X_train = np.load('train_images.npy')
X_train = rgb2gray(X_train)
# we found a better way to normalize, using BatchNormalization() layer
# X_train = X_train/255  # normalizing data
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
X_test = rgb2gray(X_test)
# we found a better way to normalize, using BatchNormalization() layer
# X_test = X_test/255  # normalizing data
y_test = np.load('test_targets.npy')

# y_test = tensorflow.keras.utils.to_categorical(y_test)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_set, y_set, test_size=0.20, random_state=42)

# select model and types of layers
# because this is a simple NN, we only added Dense layers
model = Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(256*2+128+32, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32+16, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# model.add(Flatten(input_shape=(100, 100)))
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(10, activation='softmax'))

# pick Adam as optimizer
custom = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)

# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# set optimizer, loss function and metrics to watch for model
model.compile(optimizer=custom,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# print a summary of the model
model.summary()

# implement early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7)

# fit greyscale images to model
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    callbacks=[callback],
                    epochs=100, batch_size=35, verbose=2)

model.evaluate(X_test,  y_test, verbose=2)

# plot loss and accuracy
plt.plot(history.history['loss'])
plt.title('Model loss/accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')

plt2 = plt.twinx()
color = 'red'
plt2.plot(history.history['accuracy'], color=color)
plt.ylabel('Accuracy')
plt2.legend(['Accuracy'], loc='upper center')
plt.show()
y_pred = model.predict(X_test)
# - plot loss and accuracy

# confusion matrix
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
ax = plt.subplot()
ax.set_title('Predicted vs Actual')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.show()
