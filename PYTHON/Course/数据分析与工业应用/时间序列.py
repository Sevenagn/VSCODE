import pandas as pd
import numpy as np
import torch 

df=pd.read_excel(r'C:\Users\seven\OneDrive\文档\Git\GitHub\VSCODE\PYTHON\Course\数据分析与工业应用\google.xlsx')

df2=df['Open']
#转换为numpy
A1=df2.to_numpy()

#将多行数据转换为需要形状
#用前5天的数据预测第6天
row_number=3019 - 6 + 1
idx=(np.tile(np.arange(0,6),(row_number,1)).T + np.arange(0,row_number)).T
A = A1[idx]
#数据分割
X = A[:,0:5]
Y = A[:,-1]

X = X.astype(np.float32)
Y = Y.astype(np.float32)

from torch.utils.data import DataLoader

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, ):
        pass

    def __getitem__(self, index):
        return X[index],Y[index]

    def __len__(self):
        return len(X)

ds=my_dataset()


from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import DataLoader
# 划分训练和测试的数据
train, test = random_split(ds, (2800, 214) )
# 设定每次读取的数量
data_loader = torch.utils.data.DataLoader(dataset=train,batch_size=200, shuffle=True)
data_loader2 = torch.utils.data.DataLoader(dataset=test,batch_size=200, shuffle=True)

import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        #两层Linear
        self.L1 = nn.Linear(5,100)
        #BatchNorm1d
        self.B1= nn.BatchNorm1d(100)
        self.L2 = nn.Linear(100,200)
        self.B2= nn.BatchNorm1d(200)
        self.L = nn.Linear(200,1)

    def forward(self, x):
        #降二维，从start_dim开始相乘
        # x = torch.flatten(x,start_dim = 1)
        g1 = self.L1(x)
        g1 = self.B1(g1)
        # f1=torch.sigmoid(g1)
        f1=torch.relu(g1)

        g2 = self.L2(f1)
        g2 = self.B2(g2)
        f2 = torch.relu(g2)
        #这行删掉，本来就是二维
        # f3 = torch.flatten(f3,start_dim = 1)

        g = self.L(f2)

        return g
    
model = MyNet()

import torchmetrics
#loss函数
lossf = nn.MSELoss()     
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  #SGD函数
#torchmetrics函数
metricsf = torchmetrics.MeanSquaredError()       

for i in range(100):
    for batchX, batchY in data_loader:
        score = model(batchX)
        score = torch.squeeze(score)
        loss = lossf(score, batchY)

        loss.backward() #求导
        optimizer.step() #梯度下降
        optimizer.zero_grad() #梯度清零

        mse = metricsf(score, batchY)    
    print(loss, metricsf.compute() ** 0.5)
    metricsf.reset()


torch.save(model, 'model21.pth')

for batchX, batchY in data_loader2:
    score = model(batchX)
    #加这段预测结果 score 的形状将与目标 batchY 相匹配
    score = torch.squeeze(score)
    mse = metricsf(score, batchY) 
      
print(metricsf.compute() ** 0.5 )



