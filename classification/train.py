
import numpy as np        
import torch

from torch import optim
from torch.autograd  import Variable
import torch.nn as nn

from model3 import Net
from loaddata import loadcifar

def train(ac_op, res_op):
    Epoch =30 
    
    trainset, trainloader, testset, testloader = loadcifar()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(outdim = 10, ac_op = ac_op, res_op = res_op)
    net.to(DEVICE)
    print("#params", sum([x.numel() for x in net.parameters()]))
    
    criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数
    criterion = criterion.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay=0.001)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_acc = []
    for epoch in range(Epoch):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss  = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total +=labels.size(0)
            correct +=(predicted == labels).sum()
        if epoch % 3 ==  0:# every 4 epoch
            scheduler.step()
        if epoch % 1 ==  0:
            print("Epoch: %d  Train Acc: %.4f"%(epoch, 100*correct/total))
        train_acc.append((100*correct/total).cpu().item())
        
    
    path = "./model_save/Net.pth"
    torch.save(net.state_dict(), path)
    
    print("----------finished training---------")
    correct = 0
    total = 0
    
    net.load_state_dict(torch.load(path))
    net.to(DEVICE)
    
    
    
    for data in testloader:
            images, labels = data
            net.eval()
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.no_grad():
                outputs = net(Variable(images))
                _, predicted = torch.max(outputs, 1)
                total +=labels.size(0)
                correct +=(predicted == labels).sum()
    print('Test Acc: %.4f'%(100*correct/total))
    
    test_acc = (100*correct/total).cpu().item()
    
    return np.array(train_acc), np.array(test_acc)