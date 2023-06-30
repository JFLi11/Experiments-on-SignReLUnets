

import gzip
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
import gzip
import pickle
import math
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from signrelu import signrelu
from pathlib import Path

WEIGHT_DECAY = 0.02
NUM_EPOCHS = 20
BATCH_SIZE = 20
LEARNING_RATE = 5e-3
DEVICE = torch.device("cuda")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1024)


def filters_normalized():
    P = torch.tensor([[1.,1.,1.,1.],
          [1.,-1.,0.,0.],
          [1.,0.,-1.,0.],
          [1.,0.,0.,-1.],
          [0.,1.,-1.,0.],
          [0.,1.,0.,-1.],
          [0.,0.,1.,-1.]
          ])
    P[0,:] = 0.5*P[0,:]
    P[1::,:] = P[1::,:]/math.sqrt(2)
    Q = P.clone().T
    Q[:, 1::] = 0.5*Q[:, 1::]
    #print(torch.mm(Q,P))
    
    filters_dec = torch.zeros((7,2,2))
    filters_rec = torch.zeros((7,2,2))
    A = torch.zeros((2,2))
    B = torch.zeros((2,2))
    for i in range(7):      
        A[0,0] = P[i, 2]
        A[0,1] = P[i, 3]
        A[1,0] = P[i, 0]
        A[1,1] = P[i, 1]
        
        B[0,0] = Q[2, i]
        B[0,1] = Q[3, i]
        B[1,0] = Q[0, i]
        B[1,1] = Q[1, i]
            
        filters_dec[i,:,:] = A
        filters_rec[i,:,:] = B
    
    return filters_dec[None, :,:].to(DEVICE), filters_rec[None,:,:].to(DEVICE)

def conv2d(image, filter_kernel):
    return F.conv2d(image, filter_kernel, stride=2)

def kron(image, filter_kernel):
    return torch.kron(image, filter_kernel)

def decomposition(image, filter_kernels):
    low_pass = conv2d(image, filter_kernels[0,0,:,:].unsqueeze(0).unsqueeze(0))
    x1 = conv2d(image, filter_kernels[0,1,:,:].unsqueeze(0).unsqueeze(0))
    x2 = conv2d(image, filter_kernels[0,2,:,:].unsqueeze(0).unsqueeze(0))
    x3 = conv2d(image, filter_kernels[0,3,:,:].unsqueeze(0).unsqueeze(0))
    x4 = conv2d(image, filter_kernels[0,4,:,:].unsqueeze(0).unsqueeze(0))
    x5 = conv2d(image, filter_kernels[0,5,:,:].unsqueeze(0).unsqueeze(0))
    x6 = conv2d(image, filter_kernels[0,6,:,:].unsqueeze(0).unsqueeze(0))
    
    high_passes = torch.cat((x1, x2, x3, x4, x5, x6), 1)
    return low_pass, high_passes

def reconstruction(low_pass, high_pass, filter_kernels):
    x0 = torch.kron(low_pass, filter_kernels[0,0,:,:].unsqueeze(0).unsqueeze(0))

    x1 = torch.kron(high_pass[:,0,:,:].unsqueeze(1), filter_kernels[0,1,:,:].unsqueeze(0))
    x2 = torch.kron(high_pass[:,1,:,:].unsqueeze(1), filter_kernels[0,2,:,:].unsqueeze(0))
    x3 = torch.kron(high_pass[:,2,:,:].unsqueeze(1), filter_kernels[0,3,:,:].unsqueeze(0))
    x4 = torch.kron(high_pass[:,3,:,:].unsqueeze(1), filter_kernels[0,4,:,:].unsqueeze(0))
    x5 = torch.kron(high_pass[:,4,:,:].unsqueeze(1), filter_kernels[0,5,:,:].unsqueeze(0))
    x6 = torch.kron(high_pass[:,5,:,:].unsqueeze(1), filter_kernels[0,6,:,:].unsqueeze(0))
    
    image = x0+x1+x2+x3+x4+x5+x6
    
    return image


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ac_op):

        super(Conv, self).__init__()
        
        self.conv_layer = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1))
        
        if ac_op == "ReLU":
            self.ac_op = nn.ReLU()
        elif ac_op == "ELU":
            self.ac_op = nn.ELU()
        elif ac_op == "LeakyReLU":
            self.ac_op = nn.LeakyReLU()
        elif ac_op == "SignReLU":
            self.ac_op = signrelu.apply
        
    def forward(self, x):        
        x = self.conv_layer(x)
        x = self.ac_op(x)
        return x
    
class ConvT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ac_op):

        super(ConvT, self).__init__()
        
        self.deconv_layer = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=1))
        
        if ac_op == "ReLU":
            self.ac_op = nn.ReLU()
        elif ac_op == "ELU":
            self.ac_op = nn.ELU()
        elif ac_op == "LeakyReLU":
            self.ac_op = nn.LeakyReLU()
        elif ac_op == "SignReLU":
            self.ac_op = signrelu.apply
        
    def forward(self, x):
        x = self.deconv_layer(x)
        x = self.ac_op(x)
        return x
    
    
class conv_layer(nn.Module):
    def __init__(self, n_kernels_1, n_kernels_2, n_kernels_3, ac_op):

        super(conv_layer, self).__init__()
        
        
        self.conv1 = torch.nn.Conv2d(1, n_kernels_1, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(n_kernels_1, n_kernels_2, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(n_kernels_2, n_kernels_3, kernel_size=2, stride=1)
        
        
        if ac_op == "ReLU":
            self.ac_op1 = nn.ReLU()
            self.ac_op2 = nn.ReLU()
            self.ac_op3 = nn.ReLU()
        elif ac_op == "ELU":
            self.ac_op1 = nn.ELU()
            self.ac_op2 = nn.ELU()
            self.ac_op3 = nn.ELU()
        elif ac_op == "LeakyReLU":
            self.ac_op1 = nn.LeakyReLU()
            self.ac_op2 = nn.LeakyReLU()
            self.ac_op3 = nn.LeakyReLU()
        elif ac_op == "SignReLU":
            self.ac_op1 = signrelu.apply
            self.ac_op2 = signrelu.apply
            self.ac_op3 = signrelu.apply
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ac_op1(x)
        
        x = self.conv2(x)
        x = self.ac_op2(x)
        
        x = self.conv3(x)
        x = self.ac_op3(x)
        
        return x
    
class deconv_layer(nn.Module):
    def __init__(self, n_kernels_1, n_kernels_2, n_kernels_3, ac_op):

        super(deconv_layer, self).__init__()
        
        self.conv1 = torch.nn.ConvTranspose2d(n_kernels_3, n_kernels_2, kernel_size=2, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(n_kernels_2, n_kernels_1, kernel_size=3, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(n_kernels_1, 1, kernel_size=3, stride=1)
        
        
        if ac_op == "ReLU":
            self.ac_op1 = nn.ReLU()
            self.ac_op2 = nn.ReLU()

        elif ac_op == "ELU":
            self.ac_op1 = nn.ELU()
            self.ac_op2 = nn.ELU()

        elif ac_op == "LeakyReLU":
            self.ac_op1 = nn.LeakyReLU()
            self.ac_op2 = nn.LeakyReLU()

        elif ac_op == "SignReLU":
            self.ac_op1 = signrelu.apply
            self.ac_op2 = signrelu.apply

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ac_op1(x)
        
        x = self.conv2(x)
        x = self.ac_op2(x)
        
        x = self.conv3(x)
        
        return x


class AWMCNN(nn.Module):

    def __init__(self, n_kernels_1, n_kernels_2, n_kernels_3, ac_op, res_op):
        super(AWMCNN, self).__init__()
        self.dec, self.rec = filters_normalized()
        
        self.conv1 = Conv(in_channels = 7, out_channels = n_kernels_1, kernel_size = 3, ac_op = ac_op)
        self.conv2 = Conv(in_channels = n_kernels_1, out_channels = n_kernels_2, kernel_size = 3, ac_op = ac_op)
        self.conv3 = Conv(in_channels = n_kernels_2, out_channels = n_kernels_3, kernel_size = 2, ac_op = ac_op)
        
        self.conv4 = ConvT(in_channels = n_kernels_3, out_channels = n_kernels_2, kernel_size = 2, ac_op = ac_op)
        self.conv5 = ConvT(in_channels = n_kernels_2, out_channels = n_kernels_1, kernel_size = 3, ac_op = ac_op)
        self.conv6 = ConvT(in_channels = n_kernels_1, out_channels = 7, kernel_size = 3, ac_op = ac_op)
        
        self.conv7 = Conv(in_channels = 7, out_channels = n_kernels_1, kernel_size = 3, ac_op = ac_op)
        self.conv8 = Conv(in_channels = n_kernels_1, out_channels = n_kernels_2, kernel_size = 3, ac_op = ac_op)
        self.conv9 = Conv(in_channels = n_kernels_2, out_channels = n_kernels_3, kernel_size = 2, ac_op = ac_op)
        
        self.conv10 = ConvT(in_channels = n_kernels_3, out_channels = n_kernels_2, kernel_size = 2, ac_op = ac_op)
        self.conv11 = ConvT(in_channels = n_kernels_2, out_channels = n_kernels_1, kernel_size = 3, ac_op = ac_op)
        self.conv12 = ConvT(in_channels = n_kernels_1, out_channels = 7, kernel_size = 3, ac_op = ac_op)
        
        
        self.conv_layer = conv_layer(n_kernels_1, n_kernels_2, n_kernels_3, ac_op = ac_op)
        
        self.deconv_layer = deconv_layer(n_kernels_1, n_kernels_2, n_kernels_3, ac_op = ac_op)
        
        self.res_op = res_op
        
        
    def forward(self, x):
        
        X = x
        x = self.conv_layer(x)
        x = self.deconv_layer(x)
        
        x0,x1 = decomposition(x, self.dec)
        x = torch.cat((x0,x1),1)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        if self.res_op:
            x4 = self.conv4(x3) + x2
            x5 = self.conv5(x4) + x1
            x = self.conv6(x5) + x
        else:
            x4 = self.conv4(x3)
            x5 = self.conv5(x4)
            x = self.conv6(x5)
        
        x7 = self.conv7(x)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        
        if self.res_op:
            x10 = self.conv10(x9) + x8
            x11 = self.conv11(x10) + x7
            x = self.conv12(x11) + x
        else:
            x10 = self.conv10(x9)
            x11 = self.conv11(x10)
            x = self.conv12(x11)
        
        
        x = reconstruction(x[:,0,:,:].unsqueeze(1), x[:,1::,:,:], self.rec)
        
        x = x + X
        
        x = self.conv_layer(x)
        x = self.deconv_layer(x)
        
        return x

PATH = './DATASET/images'
with gzip.open(PATH, 'rb') as f:
    GRAYIMAGES = pickle.load(f)

def load_images(images, rate):
    IMAGES = images
    
    test_images_labels = torch.from_numpy(IMAGES["images"][:, :, :, :].astype(np.float32))
    shape = IMAGES["images"][:, :, :, :].shape
    
    a = IMAGES["images"].copy()
    for i in range(shape[0]):
        max_val = np.max(IMAGES["images"][i, :, :, :])
        a[i, :, :, :] = IMAGES["images"][i, :, :, :] + max_val * rate * np.random.randn(shape[1], shape[2], shape[3])
        
    test_set = torch.from_numpy(a[:, :, :, :].astype(np.float32))
    test_dataset = data_utils.TensorDataset(test_set, test_images_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print('grey norm_test:  %e'%( torch.norm(test_images_labels-test_set, p='fro') ))
    
    return test_loader, test_dataset

def load_caltech101(caltech101sp, batch_size, rate):
    DATASET = {"train":{"images":caltech101sp[1000::,:,:,:]}, "test": {"images":caltech101sp[0:1000,:,:,:]}}  
    train_orignal = torch.from_numpy(DATASET["train"]["images"][:, :, :, :].astype(np.float32))
    test_orignal = torch.from_numpy(DATASET["test"]["images"][:, :, :, :].astype(np.float32))    
    
    shape = DATASET["train"]["images"][:, :, :, :].shape
    shape1 = DATASET["test"]["images"][:, :, :, :].shape
    
    noisy_train = np.zeros(shape)
    noisy_test = np.zeros(shape1)
    for i in range(shape[0]):
        max_val = np.max(DATASET["train"]["images"][i, :, :, :])
        noisy_train[i, :, :, :] = DATASET["train"]["images"][i, :, :, :] + max_val * rate * np.random.randn(shape[1], shape[2], shape[3])
    
    for i in range(shape1[0]):
        max_val = np.max(DATASET["test"]["images"][i, :, :, :])
        noisy_test[i, :, :, :] = DATASET["test"]["images"][i, :, :, :] + max_val * rate * np.random.randn(shape[1], shape[2], shape[3])
        
    train_set = torch.from_numpy(noisy_train.astype(np.float32))
    test_set = torch.from_numpy(noisy_test.astype(np.float32))

    
    print('norm_train: %e'%( torch.norm(train_orignal-train_set,p='fro') ))
    print('norm_test:  %e'%( torch.norm(test_orignal-test_set, p='fro') ))
    
    train_dataset = data_utils.TensorDataset(train_set, train_orignal)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = data_utils.TensorDataset(test_set, test_orignal)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, test_loader, train_dataset, test_dataset

def PSNR(outputs, labels):
    shape = labels.shape
    error = torch.norm(outputs -  labels, p='fro')
    error = error ** 2 / (shape[0] * shape [1])
    error = 10 * math.log( ( (torch.max(torch.abs(labels)).item())**2 )/error ,10) #ADD ABS
    return error


def train(n_kernels_1, n_kernels_2, n_kernels_3, ac_op, res_op, rate, train_loader, test_loader, train_dataset, test_loader_images):

    classifier = AWMCNN(n_kernels_1, n_kernels_2, n_kernels_3, ac_op, res_op)
    classifier.to(DEVICE)

    print("#params", sum([x.numel() for x in classifier.parameters()]))

    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay= WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(NUM_EPOCHS):
        losses = 0
        num = 0
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)     
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses = losses + loss.item()
            num = num + 1

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
                loss.item()), end="")
        scheduler.step()    
        train_losses.append(losses/num)
        
        print("")
        error = 0
        total = 0
        psnro = 0
        for i, (images, labels) in enumerate(test_loader):
            classifier.eval()
            shape = images.shape
            
            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs= classifier(images)
                for j in range(shape[0]):
                    error = error + PSNR(outputs[j,0,:,:], labels[j,0,:,:])
                    if epoch is (NUM_EPOCHS-1):
                        psnro = psnro + PSNR(images[j,0,:,:], labels[j,0,:,:])
                    total = total + 1
        print('Total: %d, PSNR: %f'%(total, error/total))
        if epoch is (NUM_EPOCHS-1):
            psnro = psnro/total
        test_losses.append(error/total)
    
    
    
    classifier.eval()
    PSNRs = []
    PSNRs_original = []
    for i, (images, labels) in enumerate(test_loader_images):
        classifier.eval()
        with torch.no_grad():
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            PSNRs_original.append(PSNR(images[0,0,:,:], labels[0,0,:,:]))
            outputs= classifier(images)
            PSNRs.append(PSNR(outputs[0,0,:,:], labels[0,0,:,:]))
    
    PSNRs.append(test_losses[-1])
    PSNRs_original.append(psnro)
    
    return np.array(PSNRs), np.array(PSNRs_original), np.array(train_losses), np.array(test_losses)
        

def Experiment_activation_n_structure(AC_OP, RES_OP):
    
    print(RES_OP)
    print(AC_OP)
            
    if RES_OP:
        path = './results/res'
        a=Path(path)
        b=a.mkdir(exist_ok=True)
        
        path = './results/res/' + AC_OP
        a=Path(path)
        b=a.mkdir(exist_ok=True)#
    else:
        path = './results/res_no'
        a=Path(path)
        b=a.mkdir(exist_ok=True)
        
        path = './results/res_no/' + AC_OP
        a=Path(path)
        b=a.mkdir(exist_ok=True)
    
    
    RATE = [0.2, 0.3, 0.5]
    iters = 5
    N_kernels_1=10
    N_kernels_2=15
    N_kernels_3=20
    
    
    
    DA = "ca"
    PATH = './DATASET/spherical_Caltech101'
    with gzip.open(PATH, 'rb') as f:
        CALTECH101 = pickle.load(f)
        
    PSNR4 = np.zeros((iters,len(RATE),10))
    PSNR4_o = np.zeros((iters,len(RATE),10))
    trainloss_epochs = np.zeros((iters,len(RATE),NUM_EPOCHS))
    testloss_epochs = np.zeros((iters,len(RATE),NUM_EPOCHS))
        
    for i in range(len(RATE)):
        Rate = RATE[i]
        Test_loader_images, _ = load_images(images = GRAYIMAGES,rate = Rate)
        
        print(DA)
        Train_loader, Test_loader, Train_dataset, _ = load_caltech101(caltech101sp = CALTECH101, batch_size = BATCH_SIZE, rate = Rate)
        
        for j in range(iters):
            print('noise: %.2f, iters%d'%(Rate,j))
            print(AC_OP)
            print(DA)
            time_start=time.time()
            PSNR4[j,i,:], PSNR4_o[j,i,:], trainloss_epochs[j,i,:], testloss_epochs[j,i,:] =train(n_kernels_1=N_kernels_1, n_kernels_2=N_kernels_2, n_kernels_3=N_kernels_3, ac_op=AC_OP, res_op=RES_OP, rate=Rate, train_loader=Train_loader, test_loader=Test_loader, train_dataset=Train_dataset, test_loader_images=Test_loader_images)
            time_end=time.time()
            print('time cost',(time_end-time_start)/60,'m')
        ##########################################
        test_mea = np.mean(testloss_epochs, axis=0)
        test_var = np.sqrt(np.var(testloss_epochs, axis=0))
        
        
        
        ###########################################
        PSNR_mean = np.mean(PSNR4, axis=0)
        PSNR_var = np.var(PSNR4, axis=0)
        PSNR00 = np.concatenate((PSNR_mean, PSNR_var),axis=0)
        
        
        columns = ['Barbara512', 'Boat', 'fingerprint', 'Hill', 'l512', 'Man', 'Mandrill', 'Text3', 'Text4', 'MNIST']
        data = pd.DataFrame(PSNR00, columns=columns)
        PATH = path + '/4%s.xlsx'%(DA)
        writer = pd.ExcelWriter(PATH)
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()
        
        PSNRo_mean = np.mean(PSNR4_o, axis=0)
        PSNRo_var = np.var(PSNR4_o, axis=0)
        PSNR_o = np.concatenate((PSNRo_mean, PSNRo_var),axis=0)
        
        columns = ['Barbara512', 'Boat', 'fingerprint', 'Hill', 'l512', 'Man', 'Mandrill', 'Text3', 'Text4', 'MNIST']
        data = pd.DataFrame(PSNR_o, columns=columns)
        PATH = path + '/4_o%s.xlsx'%(DA)
        writer = pd.ExcelWriter(PATH)
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()
        
#                
if __name__ == '__main__':
    RES_OP = [True]
    AC_OP = ['SignReLU','ReLU','LeakyReLU', 'ELU']
    
    for i in range(len(RES_OP)):
        for j in range(len(AC_OP)):
            
            Experiment_activation_n_structure(AC_OP = AC_OP[j], RES_OP = RES_OP[i])
