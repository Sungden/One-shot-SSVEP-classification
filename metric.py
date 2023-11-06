from __future__ import division
from __future__ import print_function
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
from scipy.io import loadmat
from sklearn import preprocessing
from scipy import signal
from config import config
import matplotlib.pyplot as plt
from audtorch.metrics.functional import pearsonr


def sliding(data,label,tw,non_overlapping_rate):
    ch=data.shape[1]# data size:(#, channels, samples)
    tw=config.T
    step = int(math.ceil(tw*non_overlapping_rate)) # step of overlapping window
    #print(step,'1111111')
    x = np.array([],dtype=np.float32).reshape(0,ch,tw) # data
    y = np.zeros([0],dtype=np.int32) # true label
    for n,freq_idx in zip(range(data.shape[0]),range(label.shape[0])):
      raw_data=data[n,:,:] #[ch,tw]
      n_samples = int(math.floor((raw_data.shape[1]-tw)/step))
      _x = np.zeros([n_samples,ch,tw],dtype=np.float32)
      _y = np.ones([n_samples],dtype=np.int32) * label[freq_idx]
      for i in range(n_samples):
        _x[i,:,:] = raw_data[:,i*step:i*step+tw]
      x = np.append(x,_x,axis=0) # [?,ch,tw], ?=runs*cl*samples
      y = np.append(y,_y)        # [?,1]

    return x, y
    
def sliding_window(data,label,ref,tw=config.T,non_overlapping_rate=0.25):
    ch=data.shape[1]# data size:(#, channels, samples)
    tw=config.T
    step = int(math.ceil(tw*non_overlapping_rate)) # step of overlapping window
    print(step,'1111111')
    x = np.array([],dtype=np.float32).reshape(0,ch,tw) # data
    y = np.zeros([0],dtype=np.int32) # true label
    z = np.array([],dtype=np.float32).reshape(0,ch,tw) # ref
    for n,freq_idx in zip(range(data.shape[0]),range(label.shape[0])):
      raw_data=data[n,:,:] #[ch,tw]
      raw_ref=ref[n,:,:] #[ch,tw]
      n_samples = int(math.floor((raw_data.shape[1]-tw)/step))
      _x = np.zeros([n_samples,ch,tw],dtype=np.float32)
      _y = np.ones([n_samples],dtype=np.int32) * label[freq_idx]
      _z = np.zeros([n_samples,ch,tw],dtype=np.float32)#ref
      for i in range(n_samples):
        _x[i,:,:] = raw_data[:,i*step:i*step+tw]
        _z[i,:,:] = raw_ref[:,i*step:i*step+tw]
      x = np.append(x,_x,axis=0) # [?,ch,tw], ?=runs*cl*samples
      y = np.append(y,_y)        # [?,1]
      z = np.append(z,_z,axis=0) # [?,ch,tw], ?=runs*cl*samples
    return x, y,z

class _CorrLayer_loss(nn.Module):
    def __init__(self):
        super(_CorrLayer_loss, self).__init__()

    def forward(self, X, T):
        # X:  n_classes
        # T:  n_classes
        X = torch.flatten(X)
        T = torch.flatten(T)
        corr = pearsonr(X, T)
        return 1 - corr


def get_Reference_Signal(list_freqs, fs, num_smpls, num_harms):
        reference_signals = []
        t = np.arange(0, (num_smpls / fs), step=1.0 / fs)
        for f in list_freqs:
                reference_f = []
                for h in range(1, num_harms + 1):
                        reference_f.append(np.sin(2 * np.pi * h * f * t)[0:num_smpls])
                        reference_f.append(np.cos(2 * np.pi * h * f * t)[0:num_smpls])
                reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals




def cca_reference(list_freqs, fs, num_smpls, num_harms):
    num_freqs = len(list_freqs)
    tidx = np.arange(0, num_smpls) / fs
    y_ref = np.zeros((num_freqs, 2*num_harms, num_smpls))

    for freq_i in range(num_freqs):
        for harm_i in range(num_harms):
            harm_i=harm_i+1
            stim_freq = list_freqs[freq_i]
            tmp=np.vstack((np.sin(2*np.pi*tidx*harm_i*stim_freq),np.cos(2*np.pi*tidx*harm_i*stim_freq)))
            y_ref[freq_i, harm_i*2-2:harm_i*2, :] = tmp

    return y_ref 

from sklearn.cross_decomposition import CCA
def find_correlation(n_components, X, Y):
    cca = CCA(n_components)
    corr = np.zeros(n_components)
    num_freq = Y.shape[0]
    result = np.zeros(num_freq)
    for freq_idx in range(0, num_freq):
        matched_X = X

        cca.fit(matched_X.T, Y[freq_idx].T)
        # cca.fit(X.T, Y[freq_idx].T)
        x_a, y_b = cca.transform(matched_X.T, Y[freq_idx].T)
        for i in range(0, n_components):
            corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
            result[freq_idx] = np.max(corr)

    return result


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    al = labels.shape[0]
    TP = 0   
    FP = 0   
    TN = 0   
    FN = 0   
    for i in range(al):
        if ((preds[i]==1)and(labels[i]==1)):
            TP += 1
        if ((preds[i]==1)and(labels[i]==0)):
            FP += 1
        if ((preds[i]==0)and(labels[i]==1)):
            FN += 1
        if ((preds[i]==0)and(labels[i]==0)):
            TN +=1
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc1 = TP/(TP+FN)
    acc2 = TN/(TN+FP)
    BN = (acc1+acc2)/2
    return correct / len(labels),acc1,acc2,BN
    
def acc10(output, labels):
    preds = output.max(1)[1].type_as(labels)
    al = labels.shape[0]
    TP = 0   
    TN = 0   
    num_1 = 0
    for i in range(al):
        if ((preds[i]==1)and(labels[i]==1)):
            TP += 1
        if ((preds[i]==0)and(labels[i]==0)):
            TN += 1
        if (labels[i]==1):
            num_1 += 1
    return num_1,TP,TN
    

def Window(Raw,raw_L):
    win_l = config.window
    win_stride = config.stride
    win_num = (1250-win_l) // win_stride
    b,_,_,_ = Raw.shape
    if (b==1):
        Win_data = torch.zeros((win_num*b,1,config.C,win_l))
        Win_label = torch.zeros(b*win_num)
    else:
        Win_data = np.zeros((win_num*b,1,config.C,win_l))
        Win_label = np.zeros(b*win_num)
    ss = 0
    for i_win in range(b):
        for j_win in range(win_num):
            Win_data[ss] = Raw[i_win,:,:,j_win*win_stride:j_win*win_stride+win_l]
            Win_label[ss] = raw_L[i_win]
            ss += 1
    return Win_data, Win_label

def stand(Raw):
    b = Raw.shape[0]
    Raw1 = np.zeros_like(Raw)
    for i_stand in range(b):
        Raw1[i_stand] = preprocessing.scale(Raw[i_stand],axis=1)
    return Raw1

def FB_stand(Raw):
    b = Raw.shape[1]
    c = Raw.shape[0]
    Raw1 = np.zeros_like(Raw)
    for i_stand in range(b):
      for j_stand in range(c):
        Raw1[j_stand,i_stand] = preprocessing.scale(Raw[j_stand,i_stand],axis=1)
    return Raw1

def filter(Raw):
    Raw1 = np.zeros_like(Raw)#(N,C,T)
    nyq = 0.5*250
    Wp = [5/nyq, 90/nyq]
    Ws = [3/nyq,92/nyq]
    N, Wn = signal.cheb1ord(Wp,Ws,3,40)
    b, a = signal.cheby1(N,0.5,Wn,'bandpass')

    for i in range(Raw.shape[0]):
        for j in range(Raw.shape[1]):
            Raw1[i,j,:] = signal.filtfilt(b,a,Raw[i,j,:],padlen=3*(max(len(b),len(a))-1)) 
    return Raw1  

def FB_filter(Raw):                  
    Raw1=np.zeros_like(Raw)
    nyq = 0.5*config.rfs
    Wp = [5/nyq, 90/nyq]
    Ws = [3/nyq,92/nyq]
    N, Wn = signal.cheb1ord(Wp,Ws,3,40)
    b, a = signal.cheby1(N,0.5,Wn,'bandpass')

    for num in range(Raw.shape[0]):
        for i in range(Raw.shape[1]):
          for j in range(Raw.shape[2]):
            Raw1[num,i,j,:] = signal.filtfilt(b,a,Raw[num,i,j,:],padlen=3*(max(len(b),len(a))-1)) 
    return Raw1  
   
def FB_filter_2(Raw):                  
    Raw1=np.zeros_like(Raw)
    nyq = 0.5*config.rfs
    Wp = [6/nyq, 90/nyq]
    Ws = [4/nyq,100/nyq] #Ws = [2/nyq,94/nyq]
    N, Wn = signal.cheb1ord(Wp,Ws,3,40)
    b, a = signal.cheby1(N,0.5,Wn,'bandpass')

    for num in range(Raw.shape[0]):
        for i in range(Raw.shape[1]):
          for j in range(Raw.shape[2]):
            Raw1[num,i,j,:] = signal.filtfilt(b,a,Raw[num,i,j,:],padlen=3*(max(len(b),len(a))-1)) 
    return Raw1 

def fft(Raw):
    Raw1=np.zeros_like(Raw)
    for num in range(Raw.shape[0]):
        for i in range(Raw.shape[1]):
          for j in range(Raw.shape[2]):
            fft_result = np.fft.fft(Raw[num,i,j,:])
            Raw1[num,i,j,:] = np.abs(fft_result)
    return Raw1     

def filter_bank(eeg):
    result = np.zeros((eeg.shape[0],eeg.shape[1],config.Nm,eeg.shape[2],eeg.shape[3]))# eeg shape:label*block*C*T
    nyq = config.rfs / 2
    passband = [5, 14, 22]
    stopband = [3, 12, 20]
    highcut_pass, highcut_stop = 90, 92

    gpass, gstop, Rp = 3, 40, 0.5
    
    for i in range(eeg.shape[0]):
      for j in range(eeg.shape[1]):
        for k in range(config.Nm):
          Wp = [passband[k] / nyq, highcut_pass / nyq]
          Ws = [stopband[k] / nyq, highcut_stop / nyq]
          [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
          [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
          data = signal.filtfilt(B, A, eeg[i,j,:,:], padlen=3 * (max(len(B), len(A)) - 1)).copy()
          result[i, j, k,:, :] = data
    return result


def Corr(Raw):
    n_sam = Raw.size(0)
    Raw = Raw.cpu()
    Raw = Raw.data.numpy()
    Raw = np.squeeze(Raw)

    fft_matrix = np.abs(fft(Raw,axis=-1))
    FFT_matrix = fft_matrix
  
    FFT_matrix = torch.FloatTensor(FFT_matrix/(config.fftn))
    FFT_matrix = FFT_matrix
    FFT_matrix = FFT_matrix.unsqueeze(1)
    return FFT_matrix

def Corr_complex(Raw):
    n_sam = Raw.size(0)
    Raw = Raw.cpu()
    Raw = Raw.data.numpy()
    Raw = np.squeeze(Raw)

    fft_real=fft(Raw,axis=-1).real
    fft_imag=fft(Raw,axis=-1).imag
    fft_matrix=np.append(fft_real[:,:,:int(fft_real.shape[2]/2)],fft_imag[:,:,int(fft_imag.shape[2]/2):],axis=2)
    
    #fft_matrix = np.abs(fft(Raw,axis=-1))
    FFT_matrix = fft_matrix
  
    FFT_matrix = torch.FloatTensor(FFT_matrix/(config.fftn))
    FFT_matrix = FFT_matrix
    FFT_matrix = FFT_matrix.unsqueeze(1)
    return FFT_matrix

        
def att_norm(att):
    mx = torch.ones((att.size(2),1))
    att_sum = torch.matmul(torch.abs(att[0]),mx)
    att_sum1 = torch.matmul(att_sum,mx.T).unsqueeze(0)
    return att/att_sum1


class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
    def __init__(self, class_num=config.num_class, smoothing=config.smooth):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num
 
    def forward(self, x, target):
        assert x.size(1) == self.class_num
        if self.smoothing == None:
            return nn.CrossEntropyLoss()(x,target)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.class_num-1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  
        
        logprobs = F.log_softmax(x,dim=-1)
        mean_loss = -torch.sum(true_dist*logprobs)/x.size(-2)  
        return mean_loss
        
class LabelSmoothingLoss1(nn.Module):
   
    def __init__(self, size=2, smoothing=config.smooth):
        super(LabelSmoothingLoss1, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
 
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        x = F.log_softmax(x,dim=-1)
        return self.criterion(x, Variable(true_dist, requires_grad=False))
        
def itr(n, p, t):
    """Compute information transfer rate (ITR).

    Definition in [1]_.

    Parameters
    ----------
    n : int
        Number of targets.
    p : float
        Target identification accuracy (0 <= p <= 1).
    t : float
        Average time for a selection (s).

    Returns
    -------
    itr : float
        Information transfer rate [bits/min]

    References
    ----------
    .. [1] M. Cheng, X. Gao, S. Gao, and D. Xu,
        "Design and Implementation of a Brain-Computer Interface With High
        Transfer Rates", IEEE Trans. Biomed. Eng. 49, 1181-1186, 2002.

    """

    if (p < 0 or 1 < p):
        raise ValueError('Accuracy need to be between 0 and 1.')
    elif (p < 1 / n):
        itr = 0
        # raise ValueError('ITR might be incorrect because accuracy < chance')
    elif (p == 1):
        itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p * np.log2(p) + (1 - p) *
               np.log2((1 - p) / (n - 1))) * 60 / t

    return itr

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
        
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, input, label ):  # input: [bs, cl]   label: [bs]
        temperature=0.17299
        pos_mask = F.one_hot(label, num_classes=config.num_class)  # [bs, cl]
        pos_sim = torch.sum(input * pos_mask, dim=1)  # [bs]
        pos_sim = torch.exp(pos_sim/temperature)  # [bs]

        neg_mask = (torch.ones_like(pos_mask) - pos_mask).bool()
        neg_sim = input.masked_select(neg_mask).view(-1, config.num_class-1)  # [bs, cl-1]
        neg_sim = torch.exp(neg_sim/temperature)  # [bs, cl-1]
        neg_sim = torch.sum(neg_sim, dim=1)  # [bs]

        return (-torch.log(pos_sim / neg_sim)).mean()    