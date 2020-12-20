import os
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load array
# X_set = load('data_set_images.npy', allow_pickle=True)
# y_set = load('data_set_targets.npy', allow_pickle=True)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_set, y_set, test_size=0.20, random_state=42)

# load data
X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')
# transform targets to target vectors
y_test = tensorflow.keras.utils.to_categorical(y_test)
y_train = tensorflow.keras.utils.to_categorical(y_train)
# get number of classes to classify
num_classes = y_train.shape[1]

# OPTIMIZED CNN

# image generator
# takes X_train as input and creates images that are:
# rotated, shifted to horizontally or vertically, zoomed in and flipped horizontally
datagen = ImageDataGenerator(
    rotation_range=6,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.15,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.15,
    zoom_range=0.05,  # set range for random zoom
    horizontal_flip=True,  # randomly flip images
)
# fit data generator to data
datagen.fit(X_train)

# choose a model and the types of layers it contains
model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',
                 strides=1, padding='same', input_shape=X_train[0].shape))
model.add(BatchNormalization())
# model.add(Conv2D(16, (3, 3), activation='sigmoid'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='sigmoid', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(256, activation='sigmoid'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# set Adam as optimizer
custom = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

# pick a loss, optimizer and metrics to watch
model.compile(loss='categorical_crossentropy',
              optimizer=custom,
              metrics=['accuracy'])
# print model summary
model.summary()

# apply early stopping based on loss
callback = tensorflow.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3)
# fitting without the data generator lower accuracy
# history = model.fit(X_train, y_train, verbose=2, epochs=10)
# train the model
# this is where you pick batch size and number of epochs
history = model.fit(datagen.flow(
    X_train, y_train, batch_size=32), callbacks=[callback], epochs=12)

# make predictions on houldout set (will return a probability distribution)
pred = model.predict(X_test)
# now pick the most likely outcome
pred = np.argmax(pred, axis=1)
# build target array
y_compare = np.argmax(y_test, axis=1)
# and calculate + print accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

# Plot training & validation loss values
# print(history.history.keys())
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

# plot confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
ax = plt.subplot()
ax.set_title('Predicted vs Actual')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.show()
