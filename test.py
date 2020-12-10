import pandas as pd
import imblearn as imb
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

# read file
path = "."
filename_read = os.path.join(path, "dataset.xls")
df = pd.read_excel(filename_read)

# drop columns
df.drop('ID', 1, inplace=True)

credit_card = df

credit_card = credit_card.reindex(np.random.permutation(credit_card.index))

result = []
for x in credit_card.columns:
    if x != 'default payment next month':
        result.append(x)

X = credit_card[result].values
y = credit_card['default payment next month'].values

sm = SMOTE(random_state=12, sampling_strategy='minority', k_neighbors=9)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1)

X_train, y_train = sm.fit_sample(X_train, y_train)

Random_Forest_model = RandomForestClassifier(
    n_estimators=100, criterion="entropy")

Random_Forest_model.fit(X_train, y_train)

y_pred = Random_Forest_model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print('The accuracy is: ', accuracy*100, '%')


# ppn = Perceptron(max_iter=1000, tol=0.001, eta0=1,
#                  warm_start=True, fit_intercept=False)

# # Use 5-fold split
# kf = KFold(10)

# fold = 1

# for train_index, validate_index in kf.split(X_train_res, y_train_res):

#     ppn.fit(X_train_res[train_index], y_train_res[train_index])
#     y_test = y_train_res[validate_index]
#     y_pred = ppn.predict(X_train_res[validate_index])
#     print(
#         f"Fold #{fold}, Training Size: {len(X_train_res[train_index])}, Validation Size: {len(X_train_res[validate_index])}")
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold += 1


# ignore

# filter data, we only keep rows with a "positive" credit_card_res
#credit_card = df[df['credit_card_res'] == 0.0]

# since all credit_card_res have value 1, we drop column
#credit_card.drop('credit_card_res', 1, inplace=True)

# reshuflle
# credit_card = credit_card.reindex(np.random.permutation(credit_card.index))

# sort
# sorted_df = credit_card.sort_values(by='default payment next month')


# zeroes_df = zeroes_df[:len(ones_df)]
# sorted_df = X_over.append(y_over)
# credit_card = sorted_df
# print(credit_card.columns)

# print(len(X_train_res), len(y_train_res))

# print(len(X_train_res), len(y_train_res))
# zeroes_df = y_train_res[y_train_res == 0.0]
# ones_df = y_train_res[y_train_res == 1.0]
# print(len(zeroes_df), len(ones_df))

# df.drop("EDUCATION", 1, inplace=True)
# df.drop('SEX', 1, inplace=True)