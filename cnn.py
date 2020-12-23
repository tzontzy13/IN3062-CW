import os
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime as dt
from sklearn.metrics import classification_report

# all sources used for tuning parameters:

# Reference 1:
# Howe, Jacob, 2020.
# Tutorial 8 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# Part 2 section.
# [Accessed 21 December 2020]

# Reference 2:
# Tensorflow Documentation, 2020.
# Early Stopping.
# [ONLINE] Available at: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# Example section.
# [Accessed 21 December 2020]

# Tensorflow Documentation, 2020.
# Adam.
# [ONLINE] Available at: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
# [Accessed 21 December 2020]

# Tensorflow Documentation, 2020.
# Dropout.
# [ONLINE] Available at: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
# [Accessed 21 December 2020]

# Tensorflow Documentation, 2020.
# ImageDataGenerator.
# [ONLINE] Available at: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# [Accessed 21 December 2020]

# Reference 3:
# Howe, Jacob, 2020.
# Tutorial 4 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# "Classification on digits" section.
# [Accessed 21 December 2020]

# Reference 4:
# Tensorflow Documentation, 2020.
# Batch Normalization.
# [ONLINE] Available at: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
# [Accessed 21 December 2020]

# Reference 5:
# Scikit-learn Documentation, 2020.
# sklearn.metrics.classification_report.
# [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# Example section.
# [Accessed 21 December 2020]

# load data
# transform it to rgb
# rescale it
X_train = np.load('train_images.npy')
X_train = rgb2gray(X_train)
X_train = np.reshape(X_train, (-1, 100, 100, 1))
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
X_test = rgb2gray(X_test)
X_test = np.reshape(X_test, (-1, 100, 100, 1))
y_test = np.load('test_targets.npy')
# transform targets to target vectors
y_test = tensorflow.keras.utils.to_categorical(y_test)
y_train = tensorflow.keras.utils.to_categorical(y_train)
# get number of classes to classify
num_classes = y_train.shape[1]

# image generator
# takes X_train as input and creates images that are:
# rotated, shifted to horizontally or vertically,
# zoomed in and flipped horizontally - reference 1
datagen = ImageDataGenerator(
    rotation_range=6,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.15,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.15,
    zoom_range=0.05,  # set range for random zoom
    horizontal_flip=True,  # randomly flip images
)

start_time = dt.datetime.now()
# fit data generator to data
datagen.fit(X_train)

# Chooses a model and the types of layers it contains.
model = Sequential()
model.add(Conv2D(16+8, kernel_size=(4, 4), activation='relu',
                 strides=1, padding='same', input_shape=X_train[0].shape))
model.add(BatchNormalization())
# model.add(Conv2D(16, (3, 3), activation='sigmoid'))
model.add(Conv2D(32+16+8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# model.add(Conv2D(32, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(32+16+8, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(Conv2D(32+16+8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128+8, activation='relu'))
# model.add(Dense(32, activation='relu'))
# When adding the dropout feature anywhere in the model, the accuracy and loss drop significantly
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Activation('softmax'))

# set Adam as optimizer
custom = tensorflow.keras.optimizers.Adam(learning_rate=0.00091)

# pick a loss, optimizer and metrics to watch
model.compile(loss='categorical_crossentropy',
              optimizer=custom,
              metrics=['accuracy'])
# print model summary
model.summary()

# apply early stopping based on loss - reference 2
callback = tensorflow.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3)
# fitting without the data generator -> lower accuracy
# history = model.fit(X_train, y_train, verbose=2, epochs=10)
# train the model
# this is where you pick batch size and number of epochs
history = model.fit(datagen.flow(
    X_train, y_train, batch_size=33), callbacks=[callback], epochs=20)

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)

score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", score)
print('\n')
print('classification report:')
print(classification_report(y_test, y_pred))
print('\n')
print("Time spent running: ", dt.datetime.now() - start_time)

# Plots training & validation loss values - reference 1
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

# Plots confusion matrix - reference 3
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
print('\n')
print(cm)
ax = plt.subplot()
ax.set_title('Predicted vs Actual')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.show()
