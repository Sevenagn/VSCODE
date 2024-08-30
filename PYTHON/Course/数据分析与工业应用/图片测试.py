import numpy as np
import matplotlib.pyplot as plt
import torch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('data_batch_1')
X = data[b'data']
Y = data[b'labels']

X = np.reshape(X, (-1, 3, 32, 32)  )
X = X.astype(np.float32)
#X = X.transpose( (0, 3, 1, 2) )

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
    dataset=ds,batch_size=10000000, shuffle=True)

import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        self.L1 = nn.Conv2d(3, 32, 7, 2, 1)
        self.B1 = nn.BatchNorm2d(32)

        self.L2 = nn.Conv2d(32, 64, 5, 2, 2)
        self.B2 = nn.BatchNorm2d(64)

        self.L3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.B3 = nn.BatchNorm2d(128)

        self.L  = nn.Linear( 6272 , 10)

    def forward(self, x):
        g1 = self.L1(x)
        g1 = self.B1(g1)
        f1 = torch.relu(g1)

        g2 = self.L2(f1)
        g2 = self.B2(g2)
        f2 = torch.relu(g2)

        g3 = self.L3(f2)
        g3 = self.B3(g3)
        f3 = torch.relu(g3)

        f3 = torch.flatten(f3, start_dim = 1)

        g = self.L(f3)
        return g
    
model = MyNet()
model = torch.load('model1.pth')
model.eval()

import torchmetrics
metricsf = torchmetrics.Accuracy(task='multiclass', num_classes=10) 

for batchX, batchY in data_loader:
    score = model(batchX)
    score = torch.squeeze(score)
    mse = metricsf(score, batchY)

print( metricsf.compute() )

import os
ims = os.listdir('D:\\Working2024\\CS\\培训2024\\测试\\')
print(ims)

im1 = plt.imread('im1.jpg')
im = im1[np.newaxis, :, :, :]
im = im.transpose( (0, 3, 1, 2) )
im = im.astype(np.float32)
im = torch.from_numpy(im)
g = model(im)
r = g.argmax().numpy()

plt.imshow(im1)
plt.text( 15, 15, str(r), color='red', fontsize=40)
plt.savefig('m1.jpg')
plt.close()

im1 = plt.imread('im2.jpg')
im = im1[np.newaxis, :, :, :]
im = im.transpose( (0, 3, 1, 2) )
im = im.astype(np.float32)
im = torch.from_numpy(im)
g = model(im)
r = g.argmax().numpy()

plt.imshow(im1)
plt.text( 15, 15, str(r), color='red', fontsize=40)
plt.savefig('m2.jpg')
plt.close()
