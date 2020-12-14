from sklearn.datasets import load_digits
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import PIL
from PIL import Image
import numpy as np
from numpy import save
from numpy import load
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

import os

# data_set = []

# images = []
# targets = []

# for i in range(10):
#     for root, dirs, files in os.walk('Dataset/' + str(i)):
#         for file in files:

#             image = Image.open('./Dataset/' + str(i) + '/' + file)
#             data = np.asarray(image)

#             if(data.shape == (100, 100, 3)):
#                 # images.append(data)
#                 # targets.append(i)

#                 data_set.append((data, i))

# data_set = np.asarray(data_set, dtype=object)

# np.random.shuffle(data_set)

# for x in data_set:
#     images.append(x[0])
#     targets.append(x[1])

# images = np.asarray(images, dtype='float64')
# targets = np.asarray(targets, dtype='float64')

# # print(images.shape)
# # print(targets.shape)

# save('data_set_images.npy', images)
# save('data_set_targets.npy', targets)

# load array
X_set = load('data_set_images.npy', allow_pickle=True)
y_set = load('data_set_targets.npy', allow_pickle=True)

# y_set = tensorflow.keras.utils.to_categorical(y_set)

X_train, X_test, y_train, y_test = train_test_split(
    X_set, y_set, test_size=0.20, random_state=42)

# num_classes = y_train.shape[1]


# UNOPTIMIZED CNN
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(4, 4),
#                  strides=1, padding='same', input_shape=(100, 100, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))

# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# print("\n")
# model.summary()


# # X_train_t = torch.from_numpy(X_train)

# history = model.fit(X_train, y_train, verbose=2, epochs=10)

# # make predictions (will give a probability distribution)
# pred_hot = model.predict(X_test)
# # now pick the most likely outcome
# pred = np.argmax(pred_hot, axis=1)
# y_compare = np.argmax(y_test, axis=1)
# # calculate accuracy
# score = metrics.accuracy_score(y_compare, pred)

# print("Accuracy score: {}".format(score))

# # print(pred_hot[:5])
# # print(pred)

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train'], loc='upper left')
# plt.show()

# mat = confusion_matrix(pred, y_compare)

# # using seaborn
# sns.heatmap(mat, square=True, annot=True, cbar=False)
# plt.xlabel('predicted value')
# plt.ylabel('true value')

# plt.show()

# OPTIMIZED CNN

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # zca_epsilon=1e-06,  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=5,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu',
                 strides=1, padding='same', input_shape=X_train[0].shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

custom = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=custom,
              metrics=['accuracy'])

# model.summary()

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=48)
# make predictions (will give a probability distribution)
pred = model.predict(X_test)
# now pick the most likely outcome
pred = np.argmax(pred, axis=1)
y_compare = np.argmax(y_test, axis=1)
# and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

# Plot training & validation loss values
print(history.history.keys())
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

# Save model and weights
model_path = os.path.join("", "saved_model")
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# SVMs

# method to plot confusion matrices
# def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.gray):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar(fraction=0.05)
#     tick_marks = np.arange(len(names))
#     plt.xticks(tick_marks, names, rotation=45)
#     plt.yticks(tick_marks, names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# todo: FIX SVM LINE 243

# build a multiclass SVM 'ovo' for one-versus-one, and
# fit the data
multi_svm = SVC(gamma='scale', decision_function_shape='ovo')
multi_svm.fit(X_train.reshape(len(X_train), -2), y_train)

y_pred = multi_svm.predict(X_test)

# put the results into a DataFrame and print side-by-side
output = pd.DataFrame(data=np.c_[y_test, y_pred])
print(output)

# calculate accuracy score and print
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# find the confusion matrix, normalise and print
# cm = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)
