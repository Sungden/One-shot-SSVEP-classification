from __future__ import division
from __future__ import print_function
from config import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
import glob
import time
import math
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import collections
import random
from tqdm import tqdm
from torch import nn
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class EEGNet(nn.Module):
    def InitialBlocks(self):
        block1 = nn.Sequential(
        nn.Conv2d(config.Nm, 96, (1,config.T),padding=(0,int(config.T/2))), #same padding
        nn.BatchNorm2d(96, False),
        nn.Conv2d(96, 96, (2*config.Nh,1), groups=96),
        nn.BatchNorm2d(96, False),
        nn.ELU(),
        nn.AvgPool2d(1, 4),
        nn.Dropout2d(0.25),
        nn.Conv2d(96, 96, (1, 16), groups=96,padding=(0,7)),
        nn.BatchNorm2d(96, False),
        nn.ELU(),
        nn.AvgPool2d(1, 8),
        nn.Dropout2d(0.25))    
        return block1
    
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Linear(inputSize, n_classes, bias=True)

    
    def CalculateOutSize(self, model, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,config.Nm, config.C, samples)
        model.eval()
        out = model(data).shape
        return out[2:]
    
    def __init__(self):
        super(EEGNet, self).__init__()
        self.blocks = self.InitialBlocks()
        self.blockOutputSize = self.CalculateOutSize(self.blocks, config.T)
        self.flatten = nn.Flatten()
        self.fc = self.ClassifierBlock(96* self.blockOutputSize[1], config.num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)  # Flatten
        x = self.fc(x)
        return x

        
###SSVEPformer
class SSVEPformer(nn.Module):
    def __init__(self):
        super(SSVEPformer, self).__init__()

        # channel combination
        self.conv1 = nn.Conv1d(config.C, 2*config.C, 1)
        self.layernorm1 = nn.LayerNorm(config.T)
       
        #CNN module
        self.layernorm2 = nn.LayerNorm(config.T)
        self.conv2 = nn.Conv1d(2*config.C, 2*config.C, 31,padding=15)
        
        #channel MPL module
        self.layernorm3 = nn.LayerNorm(config.T)
        self.fc1=nn.Linear(config.T,config.T)
        
        #MLP head
        self.flatten=nn.Flatten()
        self.fc2=nn.Linear(2*config.C*config.T,6*config.num_class)
        self.layernorm4=nn.LayerNorm(6*config.num_class)
        self.fc3=nn.Linear(6*config.num_class,config.num_class)
        
    def forward(self, x):
        # channel combination
        x = self.conv1(x)
        x = self.layernorm1(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        
        ### 1st encoder###
        #CNN module
        y = self.layernorm2(x)
        x = self.conv2(y)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y
        #channel MLP module
        y2 = self.layernorm3(x)
        x = self.fc1(y2)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y2
        
        ###2nd encoder###
        #CNN module
        y = self.layernorm2(x)
        x = self.conv2(y)
        x = self.layernorm2(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y
        #channel MPL module
        y2 = self.layernorm3(x)
        x = self.fc1(y2)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x=x+y2

        #MLP head
        x=self.flatten(x)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = self.layernorm4(x)
        x = F.gelu(x)
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
    
        return x


###convca        
## correlation analysis layer
class _CorrLayer(nn.Module):
    def __init__(self):
        super(_CorrLayer, self).__init__()

    def forward(self, X, T):
        # X: n_batch, 1, 1, n_samples
        # T: n_batch, 1, n_classes, n_samples
        T = torch.swapaxes(T, -1, -2)
        corr_xt = torch.matmul(X, T)  # n_batch, 1, 1, n_classes
        corr_xx = torch.sum(torch.square(X), -1, keepdim=True)
        corr_tt = torch.sum(torch.square(T), -2, keepdim=True)
        corr = corr_xt / (torch.sqrt(corr_xx) * torch.sqrt(corr_tt))
        return corr

# define signal-CNN
class SignalCNN(nn.Module):
    def __init__(self):
        super(SignalCNN, self).__init__()
        self.conv11 = nn.Conv2d(config.nfb, 16, kernel_size=(config.C,9),padding=(3,4))# same padding  same padding=(kernel_size-1)/2
        self.conv12 = nn.Conv2d(16, 1, kernel_size=(config.C,1),padding=(4,0))# same padding
        self.conv13 = nn.Conv2d(1, 1, kernel_size=(config.C,1))
        self.dropout1 = nn.Dropout(p=0.75)
        #self.NL = NonLocalBlock(channel=16)

    def forward(self, input):
        x = self.conv11(input)
        ####NL attention block
        #x= self.NL(x)     # NL attention has no benefits
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.dropout1(x)
        return x


# define reference-CNN
class ReferenceCNN(nn.Module):
    def __init__(self):
        super(ReferenceCNN, self).__init__()
        self.conv21 = nn.Conv2d(config.C, config.num_class, kernel_size=(1,config.C),padding=(0,4)) # same padding=(kernel_size-1)/2
        self.conv22 = nn.Conv2d(config.num_class, 1, kernel_size=(1,config.C),padding=(0,4))
        self.dropout2 = nn.Dropout(p=0.15)
        
        #self.NL = NonLocalBlock(channel=40)

    def forward(self, input):
        x = self.conv21(input)
        x = self.conv22(x)
        x = self.dropout2(x)
        return x
        
class ConvCA(nn.Module):
    def __init__(self):
        super(ConvCA, self).__init__()
        self.signal_cnn = SignalCNN()
        self.reference_cnn = ReferenceCNN()
        self.corr = _CorrLayer()
        self.fc = nn.Linear(config.num_class, config.num_class)
        self.fc1 = nn.Linear(config.T, config.num_class)
        
    def forward(self, x1, x2):
        x1 = self.signal_cnn(x1)
        x2 = self.reference_cnn(x2)
        x1 = self.fc1(x1.squeeze())
        x1=torch.unsqueeze(x1,dim=1)
        x1=torch.unsqueeze(x1,dim=1)
        corr=corr+x1
        corr=corr.reshape(-1, corr.size(1) * corr.size(2)*corr.size(3))
        out = self.fc(corr)
        return out
            
class DNN_LST(nn.Module):
    def InitialBlocks(self):
        block1 = nn.Sequential(
        nn.Conv2d(config.num_class, 1, kernel_size=1),
        nn.Conv2d(1, 120, kernel_size=(2*config.Nh, 1), bias=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2), bias=True),
        nn.Dropout2d(0.1),
        nn.ReLU(),
        nn.Conv2d(120, 120, kernel_size=(1, 10), padding=(0, 4), bias=True),
        nn.Dropout2d(config.dropout))    
        return block1
    
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Linear(inputSize, n_classes, bias=True)

    
    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, config.num_class, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]
    
    def __init__(self):
        super(DNN_LST, self).__init__()
        self.blocks = self.InitialBlocks()
        self.blockOutputSize = self.CalculateOutSize(self.blocks, 2*config.Nh, config.T)
        self.flatten = nn.Flatten()
        self.fc = self.ClassifierBlock(120* self.blockOutputSize[1], config.num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)  # Flatten
        x = self.fc(x)
        return x        
      


class DNN(nn.Module):
    def InitialBlocks(self):
        block1 = nn.Sequential(
        nn.Conv2d(config.Nm, 1, kernel_size=1),
        nn.Conv2d(1, 120, kernel_size=(config.C, 1), bias=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(120,120, kernel_size=(1, 2), stride=(1, 2), bias=True),
        nn.Dropout2d(0.1),
        nn.ReLU(),
        nn.Conv2d(120, 120, kernel_size=(1, 10), padding=(0, 4), bias=True),
        nn.Dropout2d(config.dropout))    
        return block1
    
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Linear(inputSize, n_classes, bias=True)

    
    def CalculateOutSize(self, model, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,config.Nm, config.C, samples)
        model.eval()
        out = model(data).shape
        return out[2:]
    
    def __init__(self):
        super(DNN, self).__init__()
        self.blocks = self.InitialBlocks()
        self.blockOutputSize = self.CalculateOutSize(self.blocks, config.T)
        self.flatten = nn.Flatten()
        self.fc = self.ClassifierBlock(120* self.blockOutputSize[1], config.num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(x)  # Flatten
        x = self.fc(x)
        return x




class MyNet(nn.Module):
    def InitialBlocks(self):
        block = nn.Sequential(
        nn.Conv2d(config.num_class, 1, kernel_size=1),
        nn.Conv2d(1, 120, kernel_size=(2*config.Nh, 1), bias=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2), bias=True),
        nn.Dropout2d(0.1),
        nn.ReLU(),
        nn.Conv2d(120, 120, kernel_size=(1, 10), padding=(0, 4), bias=True),
        nn.Dropout2d(config.dropout))    
        return block
        
    def InitialBlocks_1(self):
        block = nn.Sequential(
        nn.Conv2d(config.num_class, 1, kernel_size=1),
        nn.Conv2d(1, 120, kernel_size=(2*config.Nh, 1), bias=True),
        nn.Dropout2d(0.1))
        return block

    def InitialBlocks_2(self):
        block = nn.Sequential(
        nn.Conv2d(config.Nm, 1, kernel_size=1),
        nn.Conv2d(1, 120, kernel_size=(config.C, 1), bias=True),
        nn.Dropout2d(0.1))    
        return block
                
    def CNN_nlock(self):        
        block = nn.Sequential(
        nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2), bias=True),
        nn.Dropout2d(0.1),
        nn.ReLU())    
        return block

    def CNN_nlock_1(self):        
        block = nn.Sequential(
        nn.Conv2d(120, 120, kernel_size=(1, 10), padding=(0, 4), bias=True),
        nn.Dropout2d(config.dropout))    
        return block

  
    def ClassifierBlock(self, inputSize, n_classes):
        return nn.Linear(inputSize, n_classes, bias=True)

    
    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, config.num_class, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    
    def __init__(self):
        super(MyNet, self).__init__()
        self.blocks = self.InitialBlocks()
        self.blocks_1 = self.InitialBlocks_1()
        self.blocks_2 = self.InitialBlocks_2()
        self.blocks_3= self.CNN_nlock()
        self.blocks_4= self.CNN_nlock_1()
        self.blockOutputSize = self.CalculateOutSize(self.blocks, 2*config.Nh, config.T)
        self.flatten = nn.Flatten()
        self.fc = self.ClassifierBlock(120* self.blockOutputSize[1], config.num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x,x1):
        x = self.blocks_1(x)#LST
        a=x
        x1= self.blocks_2(x1)#Non LST
        b=x1
        x_m1=self.blocks_3(a+b)
        x = self.blocks_3(x)
        c=x
        x1= self.blocks_3(x1)
        d=x1
        x_m2=self.blocks_4(c+d+x_m1)
        x = self.blocks_4(x)
        e=x
        x1= self.blocks_4(x1)
        f=x1     
        x_m=self.flatten(x_m2+e+f)
        xm = self.fc(x_m)
        x = self.flatten(x)  # Flatten
        x = self.fc(x)
        x1 = self.flatten(x1)  # Flatten
        x1 = self.fc(x1)
        return x,x1,xm



        





