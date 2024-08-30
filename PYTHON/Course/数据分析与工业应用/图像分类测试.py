import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\PYTHON\Course\数据分析与工业应用\test0.csv')
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
    def __init__(self,X,Y ):
        self.X,self.Y=X,Y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)

data_loader = torch.utils.data.DataLoader(dataset=my_dataset(X,Y),batch_size=100000, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.L = nn.Linear(32*27*27,10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = torch.flatten(out, start_dim = 1)
        out = self.L(out)
        return out
 
model = MyNet()
model=torch.load(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\model11.pth')
import torchmetrics
metrics = torchmetrics.Accuracy(task='multiclass', num_classes=10)
# model=torch.load(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\model9.pth')

torch.save(model, 'model10.pth')

for batchX, batchY in data_loader:
    score = model(batchX)
    metrics(score, batchY)
    print(metrics.compute())
    metrics.reset()

#取几张图片，让模型分类，把分类的结果写在图片上，图片保存成文件
YP = score.argmax(axis=1)
X = X.transpose( (0,2,3,1) )
X = X * 255
X = X.astype(np.int32)
for i in range(10):
    plt.imshow(X[i], cmap='gray')
    plt.text(0, 0, str(YP[i].item()), fontsize=30, color='red')
    plt.savefig(str(i) + '.png')
    plt.close()
 
print(Y[:10])
print(YP[:10])
