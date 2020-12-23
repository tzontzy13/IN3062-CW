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
import datetime as dt

# all sources used for tuning parameters:

# Reference 1:
# Scikit-learn Documentation, 2020.
# sklearn.metrics.classification_report.
# [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# Example section.
# [Accessed 21 December 2020]

# Reference 2:
# Howe, Jacob, 2020.
# Tutorial 4 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# "Classification on digits" section.
# [Accessed 21 December 2020]

# Reference 3:
# Scikit-image.org Documentation, 2020.
# skimage.color.rgb2gray()
# [ONLINE] Available at: https://scikit-image.org/docs/dev/api/skimage.color.html .
# Example section.
# [Accessed 21 December 2020]

# sklearn Documentation, 2020.
# KNeighborsClassifier.
# [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# [Accessed 21 December 2020]

# sklearn Documentation, 2020.
# StandardScaler.
# [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# [Accessed 21 December 2020]

# load array
X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')

# added a scaler for preprocessing of data
scaler = preprocessing.StandardScaler()

# transform train images from rgb to greyscale
# flatten them
# apply scaler to images - reference 3
X_train = rgb2gray(X_train)
# tried to apply sobel filter, but accuracy drops
# for i in range(len(X_train)):
#     X_train[i] = sobel(X_train[i])
X_train = np.reshape(X_train, (X_train.shape[0], 10000))
scaler.fit(X_train)
scaler.transform(X_train)

# transform test images from rgb to greyscale
# flatten them
# apply scaler to images - reference 3
X_test = rgb2gray(X_test)
# tried to apply sobel filter, but accuracy drops
# for i in range(len(X_test)):
#     X_test[i] = sobel(X_test[i])
X_test = np.reshape(X_test, (X_test.shape[0], 10000))
scaler.fit(X_test)
scaler.transform(X_test)

start_time = dt.datetime.now()
# find the optimal K - number of neighbours - for this dataset
best_k = 0
best_a = 0

possible_k = np.arange(6, 14, 3)
for k in possible_k:

    model = KNeighborsClassifier(n_neighbors=k, weights='distance', 
                                algorithm='ball_tree', leaf_size=15, p=2)
    model.fit(X_train, y_train)

    # evaluate the model and update the accuracies list
    score = model.score(X_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    if(best_a < score):
        best_a = score
        best_k = k

print('\n')
# apply model and tune parameters
model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', 
                                algorithm='ball_tree', leaf_size=30, p=2)
model.fit(X_train, y_train)

# print accuracy on test data (holdout set)
score = model.score(X_test, y_test)
print("best k=%d, accuracy=%.2f%%" % (best_k, score * 100))
print('\n')
# predict on test data
# used in developing the confusion matrix and classification report
predictions = model.predict(X_test)
print("Time spent running: ", dt.datetime.now() - start_time)
print('\n')
print(classification_report(y_test,predictions))

cm = confusion_matrix(predictions, y_test)
print('confusion matrix:')
print(cm)

# plot confusion matrix - reference 2
ax = plt.subplot()
ax.set_title('KNN')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt='g')
plt.xlabel('Predicted labels', axes=ax)
plt.ylabel('True labels', axes=ax)
plt.title('knn - Confusion Matrix')
plt.show()