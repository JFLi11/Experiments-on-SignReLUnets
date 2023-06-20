
import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import time

import torchvision
import matplotlib.pyplot as plt #用于显示图片
import random
from signrelu import signrelu

def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True

setup_seed(0)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def func(x, noise):
    shape = x.shape
    y = np.zeros((shape[0],1))
    for i in range(shape[0]):    
        # y[i,0] = np.linalg.norm(x[i,:])
        y[i,0] = np.sum(x[i,:])
    f = 0.5*(2 - 2*y + 0.05*y**3)/(1 + 0.5*y**2)
    if noise:
        f = f + 0.5*np.random.randn(f.shape[0], f.shape[1]) 
    return f

# noised data
x = np.random.uniform(-10,10,(1000 ,1))
f = func(x, True)
plt.scatter(x, f, alpha=0.3)
# plt.plot(x,f)
plt.show()

# ground truth
x = np.random.uniform(-10,10,(1000 ,1))
f = func(x, False)
plt.scatter(x, f, alpha=0.3, c='r')
# plt.plot(x,f)
plt.show()

def load_data(num_data, dim, batch_size):
    DATASET_train = np.random.uniform(-10,10,(num_data ,dim))
    DATASET_test = np.random.uniform(-10,10,(2000 ,dim))
        
    train_set = torch.from_numpy(DATASET_train.astype(np.float32))
    test_set = torch.from_numpy(DATASET_test.astype(np.float32))
    
    train_label = func(DATASET_train, True)
    test_label = func(DATASET_test, False)
    train_label = torch.from_numpy(train_label.astype(np.float32))
    test_label = torch.from_numpy(test_label.astype(np.float32))
    
    
    train_dataset = data_utils.TensorDataset(train_set, train_label)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = data_utils.TensorDataset(test_set, test_label)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


########################################################################
#                        模型建立                                       #
########################################################################


class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim, ac):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1, bias=True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2, bias=True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3, bias=True))
        
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
        
        if ac == "ReLU":
            
            self.a1 = nn.ReLU()
            self.a2 = nn.ReLU()
            self.a3 = nn.ReLU()
            
        elif ac == "ELU":

            self.a1 = nn.ELU()
            self.a2 = nn.ELU()
            self.a3 = nn.ELU()
            
        elif ac == "SignReLU":
                
            self.a1 = signrelu.apply
            self.a2 = signrelu.apply
            self.a3 = signrelu.apply
            
        elif ac == "LeakyReLU":
            
            self.a1 = nn.LeakyReLU()
            self.a2 = nn.LeakyReLU()
            self.a3 = nn.LeakyReLU()
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.a1(x)
        x = self.layer2(x)
        x = self.a2(x)
        x = self.layer3(x)
        x = self.a3(x)
        x = self.layer4(x)

        return x




###################################################################################
#                             模型训练与测试                                      #
###################################################################################
def train(NUM_DATA, DIM, AC):
    
    #print(AC)
    BATCH_SIZE = 100
    data_loader_train, data_loader_test, dataset_train, dataset_test = load_data(num_data = NUM_DATA, dim = DIM, batch_size = BATCH_SIZE)
    
    
    learning_rate = 1e-4
    NUM_EPOCHS = 50
    
    model = SimpleNet(DIM, 100, 100, 100, 1, AC).to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        losses = 0
        num = 0
        for i, (x, y) in enumerate(data_loader_train):
            model.train()

            x = x.to(DEVICE)     
            y = y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses = losses + loss.item()
            num = num + 1

        train_losses.append(losses/num)
    
    num=0
    losses=0
    for i, (x, y) in enumerate(data_loader_test):
        model.eval()
        
        with torch.no_grad():
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            outputs= model(x)
            loss = criterion(outputs, y)
            losses = losses + loss.item()
            num = num+1
    print('Test loss: %.8f'%(losses/num))
    test_losses = losses/num
    
    return train_losses, test_losses
    



def exp(AC):
    
    trails = 10
    
    dim = [50,100,1000]
    
    samples = [100,500,1000,1500,2000,2500,3000,4000, 5000, 6000,7000,8000,9000,10000]
    test_losses = np.zeros((trails, len(dim), len(samples)))
    
    for i in range(trails):
        time_start=time.time()
        for j in range(len(dim)):
            
            for k in range(len(samples)):
                print('iters %d, %s, dim %d, samples %d'%(i, AC, dim[j], samples[k]))
                _, b = train(NUM_DATA=samples[k], DIM=dim[j], AC=AC)
                test_losses[i,j,k] = b
        time_end=time.time()
        print('time cost',(time_end-time_start)/60,'m')
            
    np.save('./results/%s.npy'%(AC),test_losses)
                



                
    

    



def plot_n_save(AC, j, jj):
    
    samples = [100,500,1000,1500,2000,2500,3000,4000, 5000, 6000,7000,8000,9000,10000]
    dim = [50,100,1000]
    
    
    data = np.load('./results/%s.npy'%(AC))
    
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    
    
    excel = np.concatenate((mean, var),axis=0)
    
    
    columns = samples
    data = pd.DataFrame(excel, columns=columns)
    PATH = './results/%s.xlsx'%(AC)
    writer = pd.ExcelWriter(PATH)
    data.to_excel(writer, 'page_1', float_format='%f')
    writer.save()
    writer.close()
    
    var = np.sqrt(var)
    
    # error bar
    for i in range(len(dim)):
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        k = np.array(samples)
        plt.errorbar(k[j:jj],mean[i,j:jj],yerr=var[i,j:jj],fmt='s',ecolor='g',color='m',elinewidth=2,capsize=4)
        plt.xlabel('Number of samples')
        plt.ylabel('MSE')
        plt.title('Nonlinearity = %s, dim = %d, '%(AC, dim[i],))
        
        plt.savefig('./images/%s%d.png'%(AC, dim[i]), dpi=400, bbox_inches='tight')
        
        plt.show()


# train and save
AC = ["ReLU", "LeakyReLU", "ELU", "SignReLU"]
j = 0
jj = 14
for i in range(len(AC)):
    exp(AC[i])
    plot_n_save(AC[i],j,jj)

# load and plot
j = 4
jj = 14
AC = ["ReLU", "LeakyReLU", "ELU", "SignReLU"]
for i in range(len(AC)):
    plot_n_save(AC[i],j,jj)
