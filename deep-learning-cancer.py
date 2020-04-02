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
    for i in range(len(X[:, j])):
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
    for i in range(len(X[:, j])):
        X[i, j] = get_mid_point(X[i, j])
