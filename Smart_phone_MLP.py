"""
Smart Phone
"""

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os


"""
Load data
"""

raw_data = pd.read_csv('smartphone_activity_dataset.csv')

data = raw_data.iloc[:, :561]
lables = raw_data.iloc[:, 561]

data = data.values
lables = lables.values


"""
Split Train, Test and Valid
"""

x_train, X_test, y_train, Y_test = train_test_split(data, lables, test_size=0.2, shaffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.2, shaffle=True)




print("END!!!")