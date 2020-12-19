import numpy as np
import pandas as pd
import pprint
from sklearn.datasets import load_digits
# from IPython.display import display, HTML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
# mnist = load_digits()
# print(mnist.data.shape)
# load array
X_set = np.load('data_set_images.npy', allow_pickle=True)
y_set = np.load('data_set_targets.npy', allow_pickle=True)

# X_train = np.load('train_images.npy')
# y_train = np.load('train_targets.npy')
# X_test = np.load('test_images.npy')
# y_test = np.load('test_targets.npy')
# y_set = tensorflow.keras.utils.to_categorical(y_set)

X_train, X_test, y_train, y_test = train_test_split(
    X_set, y_set, test_size=0.20, random_state=42)

# print(X_train.shape)
X_train = rgb2gray(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], 10000))
X_test = rgb2gray(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 10000))
kVals = np.arange(6, 14, 1)

for k in kVals:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,
              y_train)

    # evaluate the model and update the accuracies list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
