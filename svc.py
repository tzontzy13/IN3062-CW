import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn import preprocessing

X = np.load('data_set_images.npy')
y = np.load('data_set_targets.npy')

X = rgb2gray(X)
X = np.reshape(X, (X.shape[0], 10000))

pca = PCA(2050)
X = pca.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.svm import SVC
model = SVC(kernel='rbf', C=15, gamma='scale', decision_function_shape='ovr')
model.fit(X_train, y_train)
y_model = model.predict(X_test)

from sklearn.metrics import accuracy_score
x = accuracy_score(y_test, y_model)
print(x * 100)