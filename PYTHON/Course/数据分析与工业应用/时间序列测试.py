import pandas as pd
import numpy as np
import torch

df = pd.read_excel('google.xlsx')

df4 = df['Date']
df4 = pd.to_datetime(df4)
df5 = df4.dt.day
A4 = df5.to_numpy()
row_number = 3019 - 9 + 1
idx = ( np.tile(np.arange(0, 9), (row_number, 1) ).T + np.arange(0, row_number) ).T
A5 = A4[idx]
A5 = A5[0:-1]

df3 = df['Volume']
A2 = df3.to_numpy()
row_number = 3019 - 9 + 1
idx = ( np.tile(np.arange(0, 9), (row_number, 1) ).T + np.arange(0, row_number) ).T
A3 = A2[idx]
A3 = A3[0:-1]

df2 = df['Open']
A1 = df2.to_numpy()
row_number = 3019 - 10 + 1
idx = ( np.tile(np.arange(0, 10), (row_number, 1) ).T + np.arange(0, row_number) ).T
A = A1[idx]
X = A[:, 0:9]
Y = A[:, -1]

X = np.concatenate( (X, A3, A5), axis=1 )

u = X.mean(axis=0)
std = X.std(axis=0)
X = (X - u) / std

X = X.astype(np.float32)
Y = Y.astype(np.float32)

from torch.utils.data import DataLoader

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, ):
        pass

    def __getitem__(self, index):
        return X[index], Y[index]

    def __len__(self):
        return len(X)
    
ds = my_dataset()

import torch

from torch.utils.data import random_split
from torch.utils.data import DataLoader

data_loader = torch.utils.data.DataLoader(
    dataset=ds,batch_size=1000000, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        self.L1 = nn.Linear(27, 200)
        self.B1 = nn.BatchNorm1d(200)

        self.L2 = nn.Linear( 200 , 400)
        self.B2 = nn.BatchNorm1d(400)
        
        self.L  = nn.Linear( 400 , 1)

    def forward(self, x):
        g1 = self.L1(x)
        g1 = self.B1(g1)
        f1 = torch.relu(g1)

        g2 = self.L2(f1)
        g2 = self.B2(g2)
        f2 = torch.relu(g2)

        g = self.L(f2)
        return g
model = MyNet()
model = torch.load('model1.pth')

import torchmetrics
metricsf = torchmetrics.MeanSquaredError() 

for batchX, batchY in data_loader:
    score = model(batchX)
    score = torch.squeeze(score)
    mse = metricsf(score, batchY)

print( metricsf.compute() ** 0.5)