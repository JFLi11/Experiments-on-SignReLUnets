##############################################################################
#                      Fully-connected NN on MNIST                           #
##############################################################################
import pandas as pd
import gzip
import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import random

import torchvision
import matplotlib.pyplot as plt

from signrelu import signrelu

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def train(AC):
    print(AC)
    
    dataset_train = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    dataset_test = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=False)
    
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=128, shuffle=True ,num_workers=0)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=128, shuffle=False ,num_workers=0)
    
    learning_rate = 1e-3
    Epoch = 20
    
    model = SimpleNet(28 * 28, 100, 100, 100, 10, AC).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    print("#params", sum([x.numel() for x in model.parameters()]))
    
    train_acc = []
    test_acc = 0
    
    for epoch in range(Epoch):
        total = 0
        num_correct = 0
        
        for data in data_loader_train:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred = out.max(1)
            num_correct += (pred==label).sum().item()
            total += label.size(0)
            
        print("Epoch: %d  Train Acc: %.4f  Total: %d"%(epoch, 100*num_correct/total, total))
        
        train_acc.append(100*num_correct/total)
        
        
        
    num_correct = 0
    total = 0
    for data in data_loader_test:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
            
        model.eval()
        
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
            
            _, pred = torch.max(out, 1)
            num_correct += (pred == label).sum().item()
            total += label.size(0)
            
    print("Test Acc: %.4f  Total: %d"%(100*num_correct/total, total))
    
    test_acc = 100*num_correct/total
    
    return np.array(train_acc), np.array(test_acc)



def exp(): 
    AC_OP = ['ReLU','SignReLU', 'ELU', 'LeakyReLU' ]
    
    Epoch = 20
    
    
    Train_acc = np.zeros((len(AC_OP),Epoch))
    Test_acc = np.zeros((1,len(AC_OP)))
    
    for i in range(len(AC_OP)):
        train_acc, test_acc = train(AC_OP[i])
        Train_acc[i,:] = train_acc
        Test_acc[0, i] = test_acc
    
    print(Train_acc.shape)
    print(Test_acc.shape)
    
    np.save('./results/mnist_train_acc.npy',Train_acc)
    
        
    columns = AC_OP
    data = pd.DataFrame(Test_acc, columns=columns)
    PATH = "./results/mnist_test_acc.xlsx"
    writer = pd.ExcelWriter(PATH)
    data.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

if __name__ == '__main__':
    exp()