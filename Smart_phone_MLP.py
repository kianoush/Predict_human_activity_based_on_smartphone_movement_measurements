"""
Smart Phone
"""

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


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

x_train, X_test, y_train, Y_test = train_test_split(data, lables, test_size=0.2, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
X_valid = torch.tensor(X_valid).float()

Y_train = torch.tensor(Y_train).float()
Y_test = torch.tensor(Y_test).float()
Y_valid = torch.tensor(Y_valid).float()

"""
Model
"""
num_hiddenl = 10
num_feachers = X_train.shape[1]
num_classes = 1
model = torch.nn.Sequential(torch.nn.Linear(num_feachers, num_hiddenl),
                           torch.nn.ReLU(),
                           torch.nn.Linear(num_hiddenl, num_classes),
                           torch.nn.Sigmoid())

"""
Loss
"""
loss = torch.nn.BCELoss()

"""
Optim
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

"""
Train
"""
num_samples_train = torch.tensor(X_train.shape[0])
num_samples_test = torch.tensor(X_test.shape[0])
num_samples_valid = torch.tensor(X_valid.shape[0])

epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred = model(X_train)
    loss_value = loss(Y_pred, Y_train)
    num_corrects = torch.sum(torch.max(Y_pred, 1)[1]==Y_train)
    acc_train = num_corrects.float() /num_samples_train.float()
    loss_value.backward()
    optimizer.step()


    Y_pred = model(X_valid)
    num_corrects = torch.sum(torch.max(Y_pred, 1)[1]==Y_valid)
    acc_valid = num_corrects.float() /num_samples_valid.float()
    print("Epoch: ", epoch, 'Train Loss: ', loss_value.item(),'Train Accurecy: ',acc_train.item(), 'VALIDATION acoreccy', acc_valid.item())


print("END!!!")