# -*- coding: utf-8 -*-
"""
date: 2020-03-29
"""

"""
data preprocessing
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_excel('breast-cancer.xls')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values
