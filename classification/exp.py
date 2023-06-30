
import gzip
import numpy as np
import pickle
import pandas as pd
import torch
import random
from train import train

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)



def exp(): 
    RES_OP = [True]
    AC_OP = ['ReLU','SignReLU', 'ELU', 'LeakyReLU' ]
    
    Epoch = 30
    
    
    Train_acc = np.zeros((len(RES_OP),len(AC_OP),Epoch))
    Test_acc = np.zeros((len(RES_OP),len(AC_OP)))
    
    for j in range(len(RES_OP)):
        for i in range(len(AC_OP)):
            print("Res or not:", RES_OP[j])
            print("Ac_op:%s"%(AC_OP[i]))
            train_acc, test_acc = train(ac_op=AC_OP[i], res_op=RES_OP[j])
            Train_acc[j,i,:] = train_acc
            Test_acc[j,i] = test_acc
    

if __name__ == '__main__':
    exp()