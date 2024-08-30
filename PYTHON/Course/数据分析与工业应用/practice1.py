import pandas as pd
import numpy as np
import torch
# import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\seven.zhou\Downloads\dataset.csv')
print(df.shape)
ds = df.to_numpy()
Y, X = np.split(ds, (1, ), axis=1 )
Y = Y.squeeze()
X = X.reshape( (-1,28,28,1) )
X = X / 255
X = X.transpose( (0,3,1,2) )
X = X.astype(np.float32)

import torch
from torch.utils.data import DataLoader, random_split
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, X,Y):
        self.X,self.Y=X,Y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)
    
train, test = random_split(my_dataset(X,Y), (0.7,0.3) )
data_loader = torch.utils.data.DataLoader(dataset=train,batch_size=1000, shuffle=True)
data_loader2 = torch.utils.data.DataLoader(dataset=test,batch_size=1000, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.bn1 = nn.BatchNorm2d(32)
        self.L = nn.Linear(32*26*26,10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = torch.flatten(out, start_dim = 1)
        out = self.L(out)
        return out

model = MyNet()
for batchX, batchY in data_loader:
    out = model(batchX)
    break

import torchmetrics
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
metrics = torchmetrics.Accuracy(task='multiclass', num_classes=10)
# model = torch.load(r'D:\Seven\VSCODE\Python\Course\DataAnalysis\model8.pth')
for i in range(100):
    for batchX, batchY in data_loader:
        score = model(batchX)
        score = torch.squeeze(score)
        loss = lossf(score, batchY)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        metrics(score, batchY)
        print(loss, metrics.compute())
    metrics.reset()

torch.save(model, r'D:\Seven\VSCODE\Python\Course\DataAnalysis\model9.pth')

for batchX, batchY in data_loader2:
    score = model(batchX)
    metrics(score, batchY)
    print(metrics.compute())
    metrics.reset()
