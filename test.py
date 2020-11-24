import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read file
path = "."
filename_read = os.path.join(path, "dataset.xls")
df = pd.read_excel(filename_read)

# drop columns
df.drop('ID', 1, inplace=True)
df.drop("EDUCATION", 1, inplace=True)
df.drop('SEX', 1 , inplace=True)

covid = df
# filter data, we only keep rows with a "positive" covid_res
#covid = df[df['covid_res'] == 0.0]

# since all covid_res have value 1, we drop column
#covid.drop('covid_res', 1, inplace=True)

# reshuflle
covid = covid.reindex(np.random.permutation(covid.index))

# sort
#sorted_df = covid.sort_values(by='age')

# print(sorted_df[0:5])
# print(sorted_df[-5:])

#print(covid.columns)

result = []
for x in covid.columns:
    if x != 'default payment next month':
        result.append(x)

X = covid[result].values
y = covid['default payment next month'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1)

Random_Forest_model = RandomForestClassifier(
    n_estimators=100, criterion="entropy", max_features='log2')

Random_Forest_model.fit(X_train, y_train)

y_pred = Random_Forest_model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print('The accuracy is: ', accuracy*100, '%')


# ppn = Perceptron(max_iter=1000, tol=0.001, eta0=0.8, warm_start=True, fit_intercept=False)

# # Use 5-fold split
# kf = KFold(10)

# fold = 1

# for train_index, validate_index in kf.split(X,y):
#     ppn.fit(X[train_index],y[train_index])
#     y_test = y[validate_index]
#     y_pred = ppn.predict(X[validate_index])
#     #print(y_test)
#     #print(y_pred)
#     #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
#     print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold += 1