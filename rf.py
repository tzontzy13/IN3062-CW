import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from skimage.color import rgb2gray

# load array
X_train = np.load('train_images.npy')
y_train = np.load('train_targets.npy')
X_test = np.load('test_images.npy')
y_test = np.load('test_targets.npy')

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

for i in range(1,100):
    Random_Forest_model = RandomForestClassifier(n_estimators=i, criterion="entropy", random_state=42)
    Random_Forest_model.fit(X_train, y_train)

    y_pred = Random_Forest_model.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)
    print(i, 'The accuracy is: ',accuracy*100,'%')