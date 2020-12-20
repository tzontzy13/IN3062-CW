import numpy as np
import pandas as pd
import pprint
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.filters import sobel
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# load array
X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')

# y_set = tensorflow.keras.utils.to_categorical(y_set)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_set, y_set, test_size=0.20, random_state=42)

# added a scaler for preprocessing of data
scaler = preprocessing.StandardScaler()

# transform train images from rgb to greyscale
# flatten them
# apply scaler to images
X_train = rgb2gray(X_train)
# tried to apply sobel filter, but accuracy drops
# for i in range(len(X_train)):
#     X_train[i] = sobel(X_train[i])
X_train = np.reshape(X_train, (X_train.shape[0], 10000))
scaler.fit(X_train)
scaler.transform(X_train)

# transform test images from rgb to greyscale
# flatten them
# apply scaler to images
X_test = rgb2gray(X_test)
# tried to apply sobel filter, but accuracy drops
# for i in range(len(X_test)):
#     X_test[i] = sobel(X_test[i])
X_test = np.reshape(X_test, (X_test.shape[0], 10000))
scaler.fit(X_test)
scaler.transform(X_test)

# find the optimal K - number of neighbours - for this dataset

# kVals = np.arange(7, 11, 2)
# for k in kVals:

#     model = KNeighborsClassifier(n_neighbors=k, weights='distance', 
#                                 algorithm='ball_tree', leaf_size=15, p=2)
#     model.fit(X_train, y_train)

#     # evaluate the model and update the accuracies list
#     score = model.score(X_test, y_test)
#     print("k=%d, accuracy=%.2f%%" % (k, score * 100))

# after multiple runs, we found the optimal K is 9
# apply model and tune parameters
model = KNeighborsClassifier(n_neighbors=9, weights='distance', 
                                algorithm='ball_tree', leaf_size=15, p=2)
model.fit(X_train, y_train)

# print accuracy on test data (holdout set)
score = model.score(X_test, y_test)
print("k=%d, accuracy=%.2f%%" % (9, score * 100))

# predict on test data
# used in developing the confusion matrix and classification report
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))

cm = confusion_matrix(predictions, y_test)
print(cm)

# plot confusion matrix
ax = plt.subplot()
ax.set_title('KNN')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.show()