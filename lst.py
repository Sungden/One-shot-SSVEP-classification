# -*- coding: utf-8 -*-
"""
Least-squares Transformation (LST).
See https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e.
"""
import numpy as np
from numpy import ndarray
from scipy.linalg import pinv
from config import config
def lst_kernel(S: ndarray, T: ndarray):
    P = T @ S.T @ pinv(S @ S.T)
    return P

def GLST(data,fs,Nh,f,phase):
    """source signal estimation using LST [1]
    [1] https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
    Parameters
    ----------
    data : ndarray-like (block,n_channel_1, n_times)
        mean signal.
    mean_target : ndarray-like (n_channel_2, n_times)
        Reference signal.
    Returns
    -------
    data_after : ndarray-like (n_channel_2, n_times)
        Source signal.
    """

    X_a = np.mean(data, axis=0)
    
    #  Generate reference signal Yf
    nChannel, nTime = X_a.shape
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((nTime, 2 * Nh))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[:, iNh * 2 + 1] = y_cos
    
    X = Yf.T    
    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X_a, T=X) #(n_channel_2,n_channel_1)
   
    return PT


def MsGLST(data,fs,Nh,f_list,phase_list):
    """source signal estimation using LST [1]
    [1] https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
    Parameters
    ----------
    data : ndarray-like (Nf, block,n_channel_1, n_times)
        mean signal.
    mean_target : ndarray-like (Nf,n_channel_2, n_times)
        Reference signal.
    Returns
    -------
    data_after : ndarray-like (n_channel_2, n_times)
        Source signal.
    """

    X = np.mean(data, axis=1)#(Nf,n_channel_1, n_times)
    X_a=np.zeros((X.shape[1],X.shape[0]*X.shape[2]))
    Y=np.zeros((2 * Nh,X.shape[0]*X.shape[2]))
    Ts = 1 / fs
    n = np.arange(X.shape[2]) * Ts
        
    for nf in range(X.shape[0]):
      X_a[:,nf*X.shape[2]:nf*X.shape[2]+X.shape[2]]=X[nf,:,:]
      Yf = np.zeros((X.shape[2], 2 * Nh))
      for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f_list[nf] * (iNh + 1) * n+(iNh + 1)*phase_list[nf])
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f_list[nf] * (iNh + 1) * n+(iNh + 1)*phase_list[nf])
        Yf[:, iNh * 2 + 1] = y_cos
        
      Y[:,nf*X.shape[2]:nf*X.shape[2]+X.shape[2]]=Yf.T

    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X_a, T=Y) #(n_channel_2,n_channel_1)
   
    return PT


def MsCLST(data,fs,Nh,f_list,phase_list):
    """source signal estimation using LST [1]
    [1] https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
    Parameters
    ----------
    data : ndarray-like (block,n_channel_1, n_times)
        mean signal.
    mean_target : ndarray-like (num_class, n_times)
        Reference signal.
    Returns
    -------
    data_after : ndarray-like (num_class, n_times)
        Source signal.
    """

    X = np.mean(data, axis=0)#(n_channel_1, n_times)
    Y=np.zeros((config.num_class,X.shape[1]))
    Ts = 1 / fs
    n = np.arange(X.shape[1]) * Ts
        
    for nf in range(X.shape[0]):
      Yf = np.zeros((X.shape[1]))
      for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f_list[nf] * (iNh + 1) * n+(iNh + 1)*phase_list[nf])
        y_cos = np.cos(2 * np.pi * f_list[nf] * (iNh + 1) * n+(iNh + 1)*phase_list[nf])
        Yf=Yf+y_sin+y_cos
        
      Y[nf,:]=Yf

    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X, T=Y) #(n_channel_2,n_channel_1)
   
    return PT

def TLST(source_data,target_data):
    """

    Parameters
    ----------
    source_data : ndarray-like (n_channel_1, n_times)
    target_data : ndarray-like (n_channel_2, n_times)

    Returns
    -------
    data_after : ndarray-like (n_channel_2, n_times)
        Source signal.
    """

    PT = lst_kernel(S=source_data, T=target_data) #(n_channel_2,n_channel_1)
   
    return PT


def CLST(data,fs,Nh,f,phase):
    """source signal estimation using LST [1]
    [1] https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
    Parameters
    ----------
    data : ndarray-like (block,n_channel_1, n_times)
        mean signal.
    mean_target : ndarray-like (n_channel_2, n_times)
        Reference signal.
    Returns
    -------
    data_after : ndarray-like (n_channel_2, n_times)
        Source signal.
    """

    X_a = np.mean(data, axis=0)    
    #  Generate reference signal Yf
    nTime = data.shape[2]
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((nTime))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf=Yf+y_sin+y_cos
    
    X = Yf   
  
       
    PT = lst_kernel(S=X_a, T=X.T)
    return PT



