from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()

X = np.load('data_set_images.npy')
y = np.load('data_set_targets.npy')

X_for_misslabel = X
y_for_misslabel = y

# all sources used for tuning parameters:

# Reference 1:
# Howe, Jacob, 2020.
# Tutorial 4 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# "3. PCA for visualization: Hand-written digits" section.
# [Accessed 21 December 2020]

# Reference 2:
# Howe, Jacob, 2020.
# Tutorial 8 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# Part 2 section.
# [Accessed 21 December 2020]

# Reference 3:
# Howe, Jacob, 2020.
# Tutorial 4 (Introduction To AI).
# [ONLINE] Available at: City University Moodle
# "Classification on digits" section.
# [Accessed 21 December 2020]

# Reference 4:
# Stack Overflow, 2020.
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)
# [ONLINE] Available at: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
# Answers section.
# [Accessed 21 December 2020]

# sklearn Documentation, 2020.
# SVC.
# [ONLINE] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# [Accessed 21 December 2020]

# Scikit-image.org Documentation, 2020.
# skimage.color.rgb2gray()
# [ONLINE] Available at: https://scikit-image.org/docs/dev/api/skimage.color.html .
# Example section.
# [Accessed 21 December 2020]

# Plots sample images with their respective labels - reference 1
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    if i == 2:
        ax.set_title('RGB inputs with labels')
        # reference 4 - astype(np.uint8)
    ax.imshow(X[i].astype(np.uint8), cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y[i]),
            transform=ax.transAxes, color='green')
plt.show()


# Plots sample images without labels - reference 2
def plot_input_rgb(data):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i == 2:
            ax.set_title('RGB inputs without labels')
        ax.imshow(data[i].astype(np.uint8))
    plt.show()


# Plots sample images after applying rgb2gray on data - reference 2
def plot_input_grey(data):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i == 2:
            ax.set_title('Grey inputs without labels')
        ax.imshow(data[i],
                  )
    plt.show()


plot_input_rgb(X)
# The 'rgb2gray' function transforms the data images from rgb shape [100,100,3]
# into a gray-scaled shape [100,100,1].
# It eliminates the last two colour variances (Green and Blue),
# the data set remaining with the first color variance, Red.
# This will be seen in the plot used after the function is used.
print('\n')
print("RGB images statistics \n mean: {:.4f} \n variance: {:.4f} \n standard deviation: {:.4f}".format(
    X.mean(), X.var(), X.std()))
X = rgb2gray(X)
print('\n')
print("Greyscale images statistics \n mean: {:.4f} \n variance: {:.4f} \n standard deviation: {:.4f}".format(
    X.mean(), X.var(), X.std()))
plot_input_grey(X)

# Reshapes the data for the processing stage
X = np.reshape(X, (X.shape[0], 10000))

# Runs the Principal Component Analysis to filter the features of the data
# by picking 2025 features of data with maximum variance. - reference 3
s = datetime.datetime.now()
print('\n')
print("RUNNING...")
pca = PCA(2025)
X = pca.fit_transform(X)

# Splits the data set into test and train data sets - reference 3
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.15)
# Initializes and fits the model with training data - reference 3
model = SVC(kernel='rbf', C=15, gamma='scale', decision_function_shape='ovr')
model.fit(X_train, y_train)
y_model = model.predict(X_test)

# Prints out the accuracy after the process stage finished
print('\n')
x = accuracy_score(y_test, y_model)
print("accuracy is: ", x * 100)
print('\n')
print("time spent computing: ", datetime.datetime.now() - s)

print('\n')
print('classification report:')
print(classification_report(y_test,y_model))

# Plots confusion matrix - reference 3
mat = confusion_matrix(y_test, y_model)
print('\n')
print("confusion matrix:")
print(mat)
sns.heatmap(mat, square=True, annot=True, cmap='Reds', cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.title('Confusion Matrix')
plt.show()

# Plots sample images while including mislabelled examples
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

# Splits the original data set, with same seed, for images
originalTrain, originalTest, Originalytrain, Originalytest = train_test_split(
    X_for_misslabel, y_for_misslabel, random_state=7)

for i, ax in enumerate(axes.flat):
    if i == 2:
        ax.set_title(
            'RGB predictions with labels, mislabelled examples included')
    ax.imshow(originalTest[i].astype(np.uint8),
              cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (y_test[i] == y_model[i]) else 'red')
plt.show()
