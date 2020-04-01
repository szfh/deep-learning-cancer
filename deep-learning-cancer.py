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
X[:, 2] = np.where(X[:, 2]==datetime.datetime(2014, 10, 1), '10-14', X[:, 2])
X[:, 2] = np.where(X[:, 2]==datetime.datetime(2019, 9, 5), '5-9', X[:, 2])
print(np.unique(X[:, 2]))
X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 5, 3), '3-5', X[:, 3])
X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 8, 6), '6-8', X[:, 3])
X[:, 3] = np.where(X[:, 3]==datetime.datetime(2019, 11, 9), '9-11', X[:, 3])
X[:, 3] = np.where(X[:, 3]==datetime.datetime(2014, 12, 1), '12-14', X[:, 3])
print(np.unique(X[:, 3]))
