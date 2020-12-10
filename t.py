import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import PIL
from PIL import Image
from numpy import save
from numpy import load
import torch
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
data_set_images = load('data_set_images.npy', allow_pickle=True)
data_set_targets = load('data_set_targets.npy', allow_pickle=True)

# np.random.shuffle(data_set)

X_set = data_set_images

y_set = data_set_targets

y_set = tensorflow.keras.utils.to_categorical(y_set)

X_train, X_test, y_train, y_test = train_test_split(X_set, y_set,test_size=0.20,random_state=42)

batch_size = 128
num_classes = y_train.shape[1]
epochs = 32

model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', strides=1, padding='same', input_shape= X_train[0].shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# X_train_t = torch.from_numpy(X_train)

history = model.fit(X_train,y_train,verbose=2,epochs=16)

pred_hot = model.predict(X_test)
pred = np.argmax(pred_hot,axis=1)
y_compare = np.argmax(y_test,axis=1) 
score = metrics.accuracy_score(y_compare, pred)

print("Accuracy score: {}".format(score))

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train'], loc='upper left')
# plt.show()

# mat = confusion_matrix(pred, y_compare)

# #using seaborn 
# sns.heatmap(mat, square=True, annot=True, cbar=False)
# plt.xlabel('predicted value')
# plt.ylabel('true value');

# plt.show()