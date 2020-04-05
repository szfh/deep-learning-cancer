# -*- coding: utf-8 -*-
"""
date: 2020-03-29
"""

"""
data preprocessing
"""

# import libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import datetime as datetime

from sklearn import preprocessing

# import dataset
dataset = pd.read_excel('breast-cancer.xls')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# encode categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.float)
# y = LabelEncoder().fit_transform(y)

# replace datetimes with strings
# X[:, 2] = np.where(X[:, 2]==datetime.datetime(2014, 10, 1), '10-14', X[:, 2])
# X[:, 2] = np.where(X[:, 2]==datetime.datetime(2019, 9, 5), '5-9', X[:, 2])
# print(np.unique(X[:, 2]))
# X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 5, 3), '3-5', X[:, 3])
# X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 8, 6), '6-8', X[:, 3])
# X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 11, 9), '9-11', X[:, 3])
# X[:, 3] = np.where(X[:, 3]==datetime.datetime(2014, 12, 1), '12-14', X[:, 3])
# print(np.unique(X[:, 3]))

# function to replace datetimes with strings
def datetime_to_string(s):
    switch={
        datetime.datetime(2019, 5, 3):'3-5',
        datetime.datetime(2019, 9, 5):'5-9',
        datetime.datetime(2019, 8, 6):'6-8',
        datetime.datetime(2019, 11, 9):'9-11',
        datetime.datetime(2014, 10, 1):'10-14',
        datetime.datetime(2014, 12, 1):'12-14',
        }
    return switch.get(s,s)

for j in [2,3]:
    # for i in range(len(X[:, j])):
    for i in range(X[:, j].shape[0]):
        X[i, j] = datetime_to_string(X[i, j])

# function to get the mid point of a range of values
def get_mid_point(n):
    n = n.split('-')
    n = [int(i) for i in n]
    n = np.mean(n)
    n = np.ceil(n)
    n = int(n)
    return(n)

for j in [0,2,3]:
    for i in range(X[:, j].shape[0]):
        X[i, j] = get_mid_point(X[i, j])

# take care of missing values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values='?',strategy='most_frequent')
missingvalues = missingvalues.fit(X)
X=missingvalues.transform(X)

# for j in [1,4,7,8]:
#     print(np.unique(X[:, j], return_counts=True))

# encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X[:, 4] = LabelEncoder().fit_transform(X[:, 4])
X[:, 6] = LabelEncoder().fit_transform(X[:, 6])
X[:, 8] = LabelEncoder().fit_transform(X[:, 8])

ct_men = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct_men.fit_transform(X))
ct_quad = ColumnTransformer([('encoder', OneHotEncoder(), [9])], remainder='passthrough')
X = np.array(ct_quad.fit_transform(X))
X = np.array(X, dtype=np.int)
y = LabelEncoder().fit_transform(y)

# avoid the dummy variable trap
X = np.delete(X,[0,5],axis=1)

# split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialise the ann
classifier = Sequential()

# add the input layer and the first hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# units = average of nodes in input+output layer ((13+1)/2)
# input_dim - compulsory for this one

# add the second hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
