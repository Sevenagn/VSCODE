import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\PYTHON\Course\数据分析与工业应用\google.xlsx')
df2=df['Open']
A1 = df2.to_numpy()

row_number=3019 - 6 + 1
idx=(np.tile(np.arange(0,6),(row_number,1)).T + np.arange(0,row_number)).T
A = A1[idx]
#数据分割
X = A[:,0:5]
Y = A[:,-1]

X = X.astype(np.float32)
Y = Y.astype(np.float32)


import torch
from torch.utils.data import DataLoader, random_split
class my_dataset(torch.utils.data.Dataset):
    def __init__(self,X,Y ):
        self.X,self.Y=X,Y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)

data_loader = torch.utils.data.DataLoader(dataset=my_dataset(X,Y),batch_size=100, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Linear(5,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.conv2 = nn.Linear(100,200)
        self.bn2 = nn.BatchNorm1d(200)
        self.L = nn.Linear(200,1)
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        fout1 = torch.relu(out1)

        out2 = self.conv2(fout1)
        out2 = self.bn2(out2)
        fout2 = torch.relu(out2)

        # out = torch.flatten(fout2, start_dim = 1)
        out = self.L(fout2)
        return out
 
model = MyNet()

import torchmetrics
# lossf = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
metrics = torchmetrics.MeanSquaredError()
# model=torch.load(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\model9.pth')

model=torch.load('model22.pth')

for batchX, batchY in data_loader:
    score = model(batchX)
    score = torch.squeeze(score)
    metrics(score, batchY)
    print(metrics.compute())
    metrics.reset()

