# -*- coding: utf-8 -*-
"""
date: 2020-03-29
"""

# =============================================================================
# Part 1 - Data Preprocessing
# =============================================================================

"""import libraries"""
import numpy as np
import pandas as pd
import datetime as datetime
import itertools

"""import dataset"""
# from google.colab import files # in colab
# uploaded = files.upload()
dataset = pd.read_excel('breast-cancer.xls')

"""create arrays of independent and dependent variables"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
Some data in the dataset is incorrectly coded as dates.
eg. '3-5' -> 3 May 19.
This loop changes datetimes to the correct range.
"""
def datetime_to_string(s):
    """function to replace datetimes with strings"""
    switch={
        datetime.datetime(2019, 5, 3):'3-5',
        datetime.datetime(2019, 9, 5):'5-9',
        datetime.datetime(2019, 8, 6):'6-8',
        datetime.datetime(2019, 11, 9):'9-11',
        datetime.datetime(2014, 10, 1):'10-14',
        datetime.datetime(2014, 12, 1):'12-14',
        }
    return switch.get(s,s)

for j in [2,3]: # columns with datetimes
    for i in range(X[:, j].shape[0]): # length of array (axis=0)
        X[i, j] = datetime_to_string(X[i, j])

"""
Data is given as a range in string format.
eg. '40-49'
Strategy: replace with the midpoint of the two values (rounded up).
eg. '40-49' -> 45
"""
def get_mid_point(n):
    """function to get the mid point of a range of values"""
    n = n.split('-')
    n = [int(i) for i in n]
    n = np.mean(n)
    n = np.ceil(n)
    n = int(n)
    return(n)

for j in [0,2,3]: # columns with ranges
    for i in range(X[:, j].shape[0]): # length of array (axis=0)
        X[i, j] = get_mid_point(X[i, j])

"""
There are missing values encoded as '?'
All are categorical variables, so the mean cannot be found.
Strategy: replace with the most frequent category
"""
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values='?',strategy='most_frequent')
missingvalues = missingvalues.fit(X)
X = missingvalues.transform(X)

"""Encode categorical data"""
from sklearn.compose import ColumnTransformer

"""2 categories"""
from sklearn.preprocessing import LabelEncoder
for j in [4,6,8]: # columns with two categories
    X[:, j] = LabelEncoder().fit_transform(X[:, j])

""">2 categories"""
from sklearn.preprocessing import OneHotEncoder
ct_men = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct_men.fit_transform(X))
ct_quad = ColumnTransformer([('encoder', OneHotEncoder(), [9])], remainder='passthrough')
X = np.array(ct_quad.fit_transform(X))

"""Change type"""
X = np.array(X, dtype=np.int64)

"""Encode dependent variable"""
y = LabelEncoder().fit_transform(y)

"""
Avoid the dummy variable trap.
Categories created by OneHotEncoder are multicollinear.
One column is dropped from each.
"""
X = np.delete(X,[0,5],axis=1)

"""
Create training and test set.
n = 286
Strategy: 80% training, 20% test to balance good training data with verification.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
Feature scaling
Variables have mean = 0 and variance = 1
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# =============================================================================
# Part 2 - Artificial Neural Network
# =============================================================================

# import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

"""Initialise the ANN"""
classifier = Sequential()

"""
Add the input layer and the first hidden layer
units = average of nodes in input and output layer ((13+1)/2)
kernel_initializer =
"""
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

"""Add the second hidden layer"""
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))

"""
Add the third hidden layer.
Adding a third layer improves prediction of the test set.
However it increases the risk of overfitting.
Adding a fourth layer does not improve performance any further.
"""
"""
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
How do we know the number of layers and their types?
This is a very hard question. There are heuristics that we can use and often the best network structure is found through a process of trial and error experimentation (I explain more about this here). Generally, you need a network large enough to capture the structure of the problem.
"""
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

"""Add the output layer"""
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

"""Compile the ANN"""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
Fit the ANN to the training set
Epochs =
Steps =
"""
epochs = 1000
batch_size = 10
classifier.fit(X_train, y_train, epochs = epochs, batch_size=batch_size)

# =============================================================================
# Part 3 - Make the predictions and evaluating the model
# =============================================================================

"""Predict the test set results"""
y_pred_raw = classifier.predict(X_test) # to evaluate the model
y_pred = (y_pred_raw > 0.5)

"""Make the confusion matrix"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = classifier.evaluate(X, y)[1]

"""Plot"""
import matplotlib.pyplot as plt
import itertools
# plt.scatter(range(y_pred.shape[0]),y_pred,marker='o',color='red',edgecolors='black',alpha=1)
# plt.scatter(range(y_test.shape[0]),y_test,marker='o',color='green',edgecolors='black',alpha=1)
# plt.title('Cancer recurrence prediction\nHidden layers = 3, Epochs = %d, Steps = %d' %(n_epochs, n_steps))
# plt.ylabel('Probability of recurrence event (%)')
# plt.text(1,0.4,'[%d,%d]\n[%d,%d]' %(cm[0][0], cm[0][1], cm[1][0], cm[1][1]),size=20)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             color='white' if cm[i, j] >= cm.max()/2 else 'black')
plt.title('Confusion matrix\nAccuracy = %0.1f%%    Epochs = %d' %(100*accuracy,epochs))
plt.xticks([0,1],['No recurrence','Recurrence'])
plt.yticks([0,1],['No recurrence','Recurrence'],rotation=90,verticalalignment='center')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()
