

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class signrelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)

        return input.clamp(min=0.) + ((1.0 - input.clamp(max=0.)) ** (-1.0)) * input.clamp(max=0.)

    @staticmethod    
    def backward(ctx, grad_output):
    
        input, = ctx.saved_tensors
    
        grad_input = grad_output.clone()
    
        grad_input[input<0.0] = grad_input[input<0.0] * (1.0 - input[input<0.0]) ** (-2.0)
            
        return grad_input





if __name__ == '__main__':
    
    # test function value
    a = np.arange(-10,10,5)
    DLU = signrelu.apply
    a = torch.from_numpy(a.astype(np.float32))
    print(a)
    b = DLU(a)
    print(b)
    
    # graph of DLU
    x = np.arange(-15,5,0.1)
    x = torch.from_numpy(x.astype(np.float32))
    
    y = DLU(x)
    plt.plot(x,y)
    plt.show()
    
    # test gradient descent
    x = np.arange(-10,10,5)
    x = torch.from_numpy(x.astype(np.float32))
    x = Variable(x, requires_grad=True)
    
    y = DLU(x)
    z = torch.sum(y)
    z.backward()
    print(x.grad)
    
    ##############################
    # test function value
    a = np.arange(-10,10,5)
    a = torch.from_numpy(a.astype(np.float32))
    print(a)
    b = DLU(a)
    print(b)
    
    # graph of DLU
    x = np.arange(-15,5,0.1)
    x = torch.from_numpy(x.astype(np.float32))
    
    y = DLU(x)
    plt.plot(x,y)
    plt.show()
    
    # test gradient descent
    x = np.arange(-10,10,5)
    x = torch.from_numpy(x.astype(np.float32))
    x = Variable(x, requires_grad=True)
    
    y = DLU(x)
    z = torch.sum(y)
    z.backward()
    print(x.grad)
    
    