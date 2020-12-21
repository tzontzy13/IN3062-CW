from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X = np.load('data_set_images.npy')
y = np.load('data_set_targets.npy')

X_for_misslabel = X
y_for_misslabel = y

# Plot 1
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].astype(np.uint8), cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y[i]),
            transform=ax.transAxes, color='green')

plt.show()
# - Plot 1

# NOISY Plot 3
def plot_input_rgb(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].astype(np.uint8))
    plt.show()

def plot_input_grey(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow( data[i], #data[i].astype(np.uint8).reshape(100, 100),
                #   cmap='binary', interpolation='nearest',
                #   clim=(0, 16))
        )
    plt.show()

plot_input_rgb(X)
X = rgb2gray(X)
plot_input_grey(X)

X = np.reshape(X, (X.shape[0], 10000))

pca = PCA(2025)
X = pca.fit_transform(X)

# Plot 2
plt.scatter(X[:, 0], X[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()
# - Plot 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

model = SVC(kernel='rbf', C=15, gamma='scale', decision_function_shape='ovr')
model.fit(X_train, y_train)
y_model = model.predict(X_test)

x = accuracy_score(y_test, y_model)
print(x * 100)

# CONFUSION MATRIX
mat = confusion_matrix(y_test, y_model)
# here seaborn is used for this larger confusion matrix
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()
# - CONFUSION MATRIX

# - MIS LABELED PLOTTING
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

# split original data, with same seed, for images
originalTrain, originalTest, Originalytrain, Originalytest = train_test_split(
    X_for_misslabel, y_for_misslabel, random_state=7)
# test_images = originalTest.reshape(-1, 45, 45)

for i, ax in enumerate(axes.flat):
    ax.imshow(originalTest[i].astype(np.uint8),
              cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (y_test[i] == y_model[i]) else 'red')
plt.show()
