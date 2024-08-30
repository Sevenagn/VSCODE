import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

df = pd.read_csv()
ds = df.to_numpy()
Y, X = np.split(ds, ( ), axis=1 )
Y = Y.squeeze()
X = X.reshape( () )
X = X / 255
X = X.transpose( () )
X = X.astype(np.float32)

import torch
from torch.utils.data import DataLoader, random_split
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, ):
        pass
    def __getitem__(self, index):
        return 
    def __len__(self):
        return 
train, test = random_split(my_dataset(), () )
data_loader = torch.utils.data.DataLoader(dataset=train,batch_size=, shuffle=True)
data_loader2 = torch.utils.data.DataLoader(dataset=test,batch_size=, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d()
        self.bn1 = nn.BatchNorm2d()
        self.L = nn.Linear()
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
optimizer = torch.optim.SGD(model.parameters(), lr=)
metrics = torchmetrics.Accuracy(task='multiclass', num_classes=0)

for i in range():
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

torch.save(model, 'model7.pth')

for batchX, batchY in data_loader2:
    score = model(batchX)
    metrics(score, batchY)
    print(metrics.compute())
    metrics.reset()
