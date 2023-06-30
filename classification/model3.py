# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:03:31 2022

@author: Dawn
"""


import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage
from torch import optim
from torch.autograd  import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from signrelu import signrelu


class classifier2(nn.Module):
    def __init__(self, indim, outdim, ac_op):
        super(classifier2,self).__init__()
        
        self.fc1 = nn.Linear(indim, outdim)
        
    def forward(self,x):
        x = self.fc1(x)
        
        return  x


class resblock(nn.Module):
    def __init__(self, in_ch, out_ch, ac_op, res_op):
        super(resblock,self).__init__()
        
        
        if ac_op == "ReLU":
            self.ac_op1 = nn.ReLU()
            self.ac_op2 = nn.ReLU()
            self.ac_op3 = nn.ReLU()
            self.ac_op4 = nn.ReLU()

        elif ac_op == "ELU":
            self.ac_op1 = nn.ELU()
            self.ac_op2 = nn.ELU()
            self.ac_op3 = nn.ELU()
            self.ac_op4 = nn.ELU()

        elif ac_op == "LeakyReLU":
            self.ac_op1 = nn.LeakyReLU()
            self.ac_op2 = nn.LeakyReLU()
            self.ac_op3 = nn.LeakyReLU()
            self.ac_op4 = nn.LeakyReLU()

        elif ac_op == "SignReLU":
            self.ac_op1 = signrelu.apply
            self.ac_op2 = signrelu.apply
            self.ac_op3 = signrelu.apply
            self.ac_op4 = signrelu.apply
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1),
                                   nn.BatchNorm2d(out_ch))
        
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1, padding=1),
                                   nn.BatchNorm2d(out_ch))
        
        self.conv3 = nn.Sequential(nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1, padding=1),
                                   nn.BatchNorm2d(out_ch))
        
        self.conv4 = nn.Sequential(nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1, padding=1),
                                   nn.BatchNorm2d(out_ch))
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        
        self.res_op = res_op
        
        
    def forward(self,x):
        x2 = self.conv1x1(x)
        
        x1 = self.conv1(x)
        x1 = self.ac_op1(x1)
        
        
        
        x1 = self.conv2(x1)
        
        if self.res_op:
            x = x1 + x2
        else:
            x = x1
        
        x1 = self.conv3(x)
        x1 = self.ac_op3(x1)
        
        
        
        x1 = self.conv4(x1)
        
        if self.res_op:
            x = x1 + x
        else:
            x = x1
        
        
        
        return  x


class Net(nn.Module):
    def __init__(self, outdim, ac_op, res_op):
        super(Net,self).__init__()
        
        self.conv1 = resblock(in_ch = 3, out_ch = 16, ac_op = ac_op, res_op = res_op) 
        self.conv2 = resblock(in_ch = 16, out_ch = 16, ac_op = ac_op, res_op = res_op) 
        self.conv3 = resblock(in_ch = 16,out_ch = 16, ac_op = ac_op, res_op = res_op) 
        self.conv4 = resblock(in_ch = 16, out_ch = 16, ac_op = ac_op, res_op = res_op) 
        
        self.classifier = classifier2(indim=16*2*2, outdim = outdim, ac_op = ac_op)
        
    def forward(self,x):
        x = F.max_pool2d(self.conv1(x),(2,2))
        x = F.max_pool2d(self.conv2(x),(2,2))
        x = F.max_pool2d(self.conv3(x),(2,2))
        x = F.avg_pool2d(self.conv4(x), (2,2))
        x = x.view(x.size()[0],-1)
        
        
        x = self.classifier(x)
        return  x

if __name__ == '__main__':
    
    model = Net(outdim=10, ac_op="ReLU", res_op=True)
    x = torch.rand(32,3,32,32)
    y = model(x)
    