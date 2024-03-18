# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:14:51 2023

@author: Yang.D
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import os
import time
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.io import loadmat
from config import config
from model import DNN,DNN_LST,MyNet
from metric import LabelSmoothingLoss, FB_stand, FB_filter, filter_bank, itr,FB_filter_2
from lst import GLST
from itertools import combinations
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from FBTDCA import generate_cca_references,FBTRCA,FBTDCA,FBMsCCA
from SAME import generate_filterbank, SAME
import torch.nn.functional as F

# GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
dist.init_process_group(backend="nccl")  # , init_method="env://", world_size=torch.cuda.device_count(),rank=local_rank)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# cpu
cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


class MyDataset(Dataset):
    def __init__(self, input1, input2, targets):
        self.input1 = input1
        self.input2 = input2
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.input1[index], self.input2[index], self.targets[index]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(41)
print('------one_shot--------')
save_model_name = 'one_shot'
is_dataset = config.Dataset

CNN=config.CNN

if is_dataset == 0:
    key_word = 'eeg'
    nCondition = 12
    rfs = 256  # sampling rate
    dataLength = 1114  # [-0.5 5.5s]
    nBlock = 15  # six blocks
    delay =0.35# 0.15  # 0.35 is better than 0.15
    latencyDelay = int(delay*rfs)  # 150ms delay 1114-1024
    list_freqs = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75,14.75]).T  # list of stimulus frequencies
    list_phase = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi  # list of stimulus phase
    name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    path1 = '/data/12JFPM/'
    index_class = range(0, config.num_class)
    channels = [0, 1, 2, 3, 4, 5, 6, 7]  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(4*rfs)
    FB_filter=FB_filter_2
    
elif is_dataset == 1:
    key_word = 'data'
    nCondition = 40
    rfs = 250  # sampling rate
    dataLength = 6 * rfs  # [-0.5 5.5s]
    nBlock = 6  # six blocks
    delay = 0.14 + 0.5  # visual latency being considered in the analysis [s]
    latencyDelay = int(delay * rfs)  # 140ms delay
    n_bands = 5  # number of sub-bands in filter bank analysis
    list_freqs = loadmat("/data/Bench/Freq_Phase.mat")['freqs'][0]
    list_phase=np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5])* np.pi 
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35']

    path1 = '/data/Bench/'
    index_class = range(0, config.num_class)
    channels = [47, 53, 54, 55, 56, 57, 60, 61,62];  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(5*rfs)

elif is_dataset == 2:
    key_word = 'data'
    nCondition = 40
    rfs = 250  # sampling rate
    # dataLength = 3 * rfs  # [-0.5 2.5s]
    nBlock = 4  # six blocks
    delay = 0.13 + 0.5  # visual latency being considered in the analysis [s]
    latencyDelay = int(delay * rfs)  # 140ms delay
    list_freqs = loadmat("/data/BETA/Freqs_Beta.mat")['freqs'][0]
    list_phase = np.array(
        [1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
         1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1]) * np.pi
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
            'S47', 'S48', 'S49', 'S50', 'S51',
            'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
            'S67', 'S68', 'S69', 'S70']
    
    path1 = '/data/BETA/'
    index_class = range(0, config.num_class)
    channels = [47, 53, 54, 55, 56, 57, 60, 61,62];  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(3*rfs)

path2 = '.mat'
PRmatrix = np.zeros(len(name))
PRmatrix_itr = np.zeros(len(name))
PRmatrix_MsCCA_1 = np.zeros(len(name))
PRmatrix_itr_MsCCA_1 = np.zeros(len(name))
PRmatrix_MsCCA_2 = np.zeros(len(name))
PRmatrix_itr_MsCCA_2 = np.zeros(len(name))
PRmatrix_final = np.zeros(len(name))
PRmatrix_itr_final = np.zeros(len(name))
num_harms = config.Nh
num = 0
for id_name in range(len(name)):
    if is_dataset == 0:
        name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    elif is_dataset == 1:
        name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
                'S32', 'S33', 'S34', 'S35']
    else:
        name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
                'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
                'S47', 'S48', 'S49', 'S50', 'S51',
                'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
                'S67', 'S68', 'S69', 'S70']
        

    name_test = [name[id_name]]
    name_train = name
    del name_train[id_name]

    name1 = name_train
    print('using',CNN,'!!!!!!')

    model_name = 'CSDuDoFN_TDCA_TRCA'

    epochs =config.epoch
    step = config.lr
    criterion = LabelSmoothingLoss()  # nn.CrossEntropyLoss()
    if CNN=='MyNet':
        model = MyNet() 
    elif CNN=='DNN':
        model = DNN() 
    else:
        model = DNN_LST()

    model = model.to(device)

    if (torch.cuda.device_count() > 1) and (dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    t_total = time.time()
    P_train = np.zeros((len(name1), config.num_class, num_harms * 2, config.C))
    P_train_1 = np.zeros((len(name1), config.num_class, config.C))

    for i in range(len(name1)):
        path = path1 + name1[i] + path2
        mat = loadmat(path)
        data_raw = mat[key_word]
        if is_dataset == 0:
            data_raw = data_raw.transpose([1, 2, 0, 3])  # [C,T,num_class,nblock]
        elif is_dataset == 2:
            data_raw = data_raw[0][0][0]
            data_raw = data_raw.transpose([0, 1, 3, 2])  # [C,T,num_class,nblock]

        a = data_raw[channels, :, :, :]
        data_raw = a[:, latencyDelay:latencyDelay + config.T, index_class, :]
        data_raw = data_raw.transpose(2, 3, 0, 1)  # label*block*C*T
        data_fb = filter_bank(data_raw)
        #data_raw = FB_filter(data_raw)
        ###GLST estimation
        data1c = np.zeros((data_raw.shape[0], data_raw.shape[1], config.num_class, 2 * config.Nh,data_raw.shape[3]))  # label*block*label*C*T
        ###Non LST
        data1cc = np.zeros((data_raw.shape[0], data_raw.shape[1], config.Nm, data_raw.shape[2],data_raw.shape[3]))  # label*block*Nm*C*T


        for cla in range(config.num_class):
            # print(data.shape,'000000')#(6, 9, 175)
            P = GLST(data_raw[cla, :, :, :], rfs, num_harms, list_freqs[cla], list_phase[cla])
            P_train[i, cla, :, :] = P
            for idx in range(data_raw.shape[0]):
                for blo in range(data_raw.shape[1]):
                    data_after = P @ data_raw[idx, blo, :, :]
                    data1c[idx, blo, cla, :, :] = data_after

        train_data_p = np.zeros((nBlock * config.num_class, data1c.shape[2], data1c.shape[3], data1c.shape[4]))
        train_data_cc = np.zeros((nBlock * config.num_class, data1cc.shape[2], data1cc.shape[3], data1cc.shape[4]))
        train_label_p = np.zeros(nBlock * config.num_class)

        for j in range(config.num_class):
            train_data_p[nBlock * j:nBlock * j + nBlock] = data1c[j]
            train_data_cc[nBlock * j:nBlock * j + nBlock] = data_fb[j]
            train_label_p[nBlock * j:nBlock * j + nBlock] = np.ones(nBlock) * j
        if (i == 0):
            train_datac = train_data_p
            train_datac_cc = train_data_cc
            train_labelc = train_label_p
        else:
            train_datac = np.append(train_datac, train_data_p, axis=0)
            train_datac_cc = np.append(train_datac_cc, train_data_cc, axis=0)
            train_labelc = np.append(train_labelc, train_label_p)
        if (dist.get_rank() == 0):
            print(train_datac.shape)  # (8160, 40, 8, 125)#GLST
            print(train_datac_cc.shape)  # (8160, 3, 9, 125)#Non LST
            print('-----------------')

    datas = train_datac
    datas = FB_stand(datas)
    datas_cc = train_datac_cc
    datas_cc = FB_stand(datas_cc)
    label = train_labelc
    print(datas.shape, datas_cc.shape, '11111111')  # (8160, 40, 8, 125) (8160, 3, 9, 125) (8160, 1, 40, 125)

    if (dist.get_rank() == 0):
        print(datas.shape)  # (8160, 40, 8, 125)

    a = np.random.permutation(datas.shape[0])
    datas = datas[a]
    datas_cc = datas_cc[a]
    label = label[a]

    num_val = config.val
    val_data = torch.FloatTensor(datas[datas.shape[0] - num_val:])
    val_data_cc = torch.FloatTensor(datas_cc[datas_cc.shape[0] - num_val:])
    val_label = torch.FloatTensor(label[datas.shape[0] - num_val:])
    datas = datas[:datas.shape[0] - num_val]
    datas_cc = datas_cc[:datas_cc.shape[0] - num_val]
    label = label[:label.shape[0] - num_val]
    train_data = torch.FloatTensor(datas)
    train_data_cc = torch.FloatTensor(datas_cc)
    train_label = torch.FloatTensor(label)

    dataset_train = MyDataset(train_data, train_data_cc, train_label)
    dataloader_train = DataLoader(dataset_train, batch_size=config.batchsize, shuffle=True)
    dataset_val = MyDataset(val_data, val_data_cc, val_label)
    dataloader_val = DataLoader(dataset_val, batch_size=config.batchsize, shuffle=True)

    ##########################Train###############################
    if (dist.get_rank() == 0):
        print(name_test[0])
    val_max = 0
    stepp_new = 0

    for i in range(epochs):
        t = time.time()
        if (i % 20 == 0 and i > 0):
            step = step * 0.8
        optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=0.01)
        train_l_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        model.train()
        for ii1, data in enumerate(dataloader_train, 0):
            inputs, inputs_cc, labels = data
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)  # [64, 1, 64, 500]
            inputs_cc = inputs_cc.to(device)
            if CNN=='MyNet': 
                outputs_label, outputs_label_1, outputs_label_m = model(inputs, inputs_cc)
                loss = criterion(outputs_label, labels.long())
                loss_1 = criterion(outputs_label_1, labels.long())
                loss_m = criterion(outputs_label_m, labels.long())
                loss_total = criterion(outputs_label + outputs_label_1 + outputs_label_m, labels.long())
                loss = (loss + loss_1 + loss_m + loss_total ) / 4
            elif CNN=='DNN':
                outputs_label= model(inputs_cc)
                loss = criterion(outputs_label, labels.long())
            else:
                outputs_label= model(inputs)
                loss = criterion(outputs_label, labels.long())

            loss.backward()
            optimizer.step()
            train_l_sum += loss.cpu().item()
            train_acc_sum += ((outputs_label).argmax(dim=1) == labels).sum().cpu().item()
            n += labels.shape[0]

        sum_0 = n - sum_1
        train_l_sum = train_l_sum / (ii1 + 1)
        BN = train_acc_sum / n

        # Validation
        val_l_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii2, data in enumerate(dataloader_val, 0):
            val_inputs, val_inputs_cc, val_labels = data
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.to(device)
            val_inputs_cc = val_inputs_cc.to(device)
            if CNN=='MyNet':  
                val_output, val_output_1, val_output_m = model(val_inputs, val_inputs_cc)
                loss_val = criterion(val_output, val_labels.long())
                loss_val_1 = criterion(val_output_1, val_labels.long())
                loss_val_m = criterion(val_output_m, val_labels.long())
                loss_total = criterion(val_output + val_output_1 + val_output_m, val_labels.long())
                loss_val = (loss_val + loss_val_1 + loss_val_m + loss_total ) / 4
            elif CNN=='DNN':
                val_output= model(val_inputs_cc)
                loss_val = criterion(val_output, val_labels.long())
            else:
                val_output= model(val_inputs)
                loss_val = criterion(val_output, val_labels.long())

            val_l_sum += loss_val.cpu().item()
            val_acc_sum += ((val_output).argmax(dim=1) == val_labels).sum().cpu().item()
            n += val_labels.shape[0]

        sum_0 = n - sum_1
        val_l_sum = val_l_sum / (ii2 + 1)
        val_BN = val_acc_sum / n

        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i + 1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                  "BN= {:.4f}".format(BN),
                  'loss_val: {:.4f}'.format(val_l_sum),
                  "val_BN= {:.4f}".format(val_BN),
                  "time: {:.4f}s".format(time.time() - t))

        if (val_BN > val_max):
            val_max = val_BN
            stepp_new = 0
            if (dist.get_rank() == 0):
                save_path = os.path.join(config.save_path, model_name, 'dataset_' + str(config.Dataset))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'stimulus_' + str(
                    config.T / rfs) + '_s_bestpath_' + save_model_name + '.pkl'))

        stepp_new = stepp_new + 1
        if (stepp_new == config.patience):
            break

    ############################testing stage#####################################
    print('the second stage starting...')
    path = path1 + name_test[0] + path2
    mat = loadmat(path)
    data_raw_test = mat[key_word]
    if is_dataset == 0:
        data_raw_test = data_raw_test.transpose([1, 2, 0, 3])
    elif is_dataset == 2:
        data_raw_test = data_raw_test[0][0][0]
        data_raw_test = data_raw_test.transpose([0, 1, 3, 2])  # [C,T,num_class,nblock]

    b = data_raw_test[channels, :, :, :]
    data_raw_test = b[:, latencyDelay:latencyDelay + config.T, index_class, :]
    data_raw_test = data_raw_test.transpose(2, 3, 0, 1)  # label*block*C*T
    
    data_raw_test_TDCA = b[:, latencyDelay:latencyDelay + config.T+int(0.1*rfs), index_class, :]
    data_raw_test_TDCA = data_raw_test_TDCA.transpose(2, 3, 0, 1)  # label*block*C*T

    acc_all_block = []
    itr_all_block = []
    acc_all_block_MsCCA_1=[]
    itr_all_block_MsCCA_1=[]
    acc_all_block_MsCCA_2=[]
    itr_all_block_MsCCA_2=[]
    acc_all_block_final = []
    itr_all_block_final = []
    # LOOC-shot
    indices = np.arange(nBlock)
    LOOC = 1
    train_pairs = list(combinations(indices, LOOC))
    cv = 0
    for train_indices in train_pairs:
        print('cross vatlidation...', str(cv), ' starting...')
        cv += 1
        test_runs = np.setdiff1d(indices, train_indices)
        train_runs = list(train_indices)

        ############################Training block#####################################
        data1c_train = data_raw_test[:, train_runs, :, :]
        data_fb_train = filter_bank(data1c_train)
        #data1c_train = FB_filter(data1c_train)
        data1c_train_=data1c_train

        data1c_train_TDCA= data_raw_test_TDCA[:, train_runs, :, :]
        #data1c_train_TDCA = FB_filter(data1c_train_TDCA)
        data1c_train_label = np.arange(config.num_class)
        
        ###GLST estimation
        data1c_train_new = np.zeros((data1c_train.shape[0], config.num_class, 2 * config.Nh, data1c_train.shape[3]))  # label*label*C*T
        P_test = np.zeros((config.num_class, num_harms * 2, config.C))
        P_test_1 = np.zeros((config.num_class, config.C))
        for cla in range(config.num_class):
            P = GLST(data1c_train[cla], rfs, num_harms, list_freqs[cla], list_phase[cla])
            P_test[cla, :, :] = P

            for idx in range(data_raw_test.shape[0]):
                data_after = P @ data1c_train[idx, :, :]
                data1c_train_new[idx, cla, :, :] = data_after

        ############################Testing block#####################################
        data1c_test = data_raw_test[:, test_runs, :, :]
        data1c_test_FB = filter_bank(data1c_test)
        #data1c_test = FB_filter(data1c_test)
        data1c_test_=data1c_test

        ###GLST
        data1c = np.zeros((data1c_test.shape[0], data1c_test.shape[1], config.num_class, 2 * config.Nh,data1c_test.shape[3]))  # label*block*label*C*T
        ###Non LST
        data1cc = np.zeros((data1c_test.shape[0], data1c_test.shape[1], config.Nm, data1c_test.shape[2],data1c_test.shape[3]))  # label*block*Nm*C*T

        for cla in range(config.num_class):
            P = P_test[cla, :, :]
            for idx in range(data1c_test.shape[0]):
                for blo in range(data1c_test.shape[1]):
                    data_after = P @ data1c_test[idx, blo, :, :]
                    data1c[idx, blo, cla, :, :] = data_after

        test_data_p = np.zeros((config.num_class * (nBlock - LOOC), data1c.shape[2], data1c.shape[3], data1c.shape[4]))
        test_data_p_cc = np.zeros((config.num_class * (nBlock - LOOC), data1cc.shape[2], data1cc.shape[3], data1cc.shape[4]))
        test_label_p = np.zeros(config.num_class * (nBlock - LOOC))
        test_data_MsCCA = np.zeros((config.num_class * (nBlock - LOOC), data1c_test_.shape[2], data1c_test_.shape[3]))
        for j in range(config.num_class):
            test_data_p[(nBlock - LOOC) * j:(nBlock - LOOC) * j + (nBlock - LOOC)] = data1c[j]
            test_data_p_cc[(nBlock - LOOC) * j:(nBlock - LOOC) * j + (nBlock - LOOC)] = data1c_test_FB[j]
            test_label_p[(nBlock - LOOC) * j:(nBlock - LOOC) * j + (nBlock - LOOC)] = np.ones(nBlock - LOOC) * j
            test_data_MsCCA[(nBlock - LOOC) * j:(nBlock - LOOC) * j + (nBlock - LOOC)] = data1c_test_[j]
            

        datas = test_data_p
        datas = FB_stand(datas)
        datas_cc = test_data_p_cc
        datas_cc = FB_stand(datas_cc)
        label = test_label_p
        
        test_data = torch.FloatTensor(datas)
        test_data_cc = torch.FloatTensor(datas_cc)
        test_label = torch.FloatTensor(label)
        #########SAME###############
        wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
        ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
        filterbank = generate_filterbank(wp,ws,srate=rfs,order=15,rp=0.5)
        filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
        Yf = generate_cca_references(list_freqs, srate=rfs, T=config.sample_length,phases=list_phase,n_harmonics = 5)
        same = SAME(fs = rfs, Nh = 5, flist = list_freqs, n_Aug = 3)# for 1 block, n_Aug is 3

        ####TRCA
        same.fit(data1c_train_ , data1c_train_label)#for TRCA and MsCCA
        X_aug, y_aug = same.augment()
        print('original shape is:',data1c_train_.shape,'augmented shape is:',X_aug.shape)
        y_train_new = np.concatenate((data1c_train_label, y_aug), axis=0)
        X_train_new = np.concatenate((np.squeeze(data1c_train_), X_aug), axis=0)#For TRCA and MsCCA
        estimator = FBTRCA(filterbank=filterbank,n_components = 1, ensemble = True,filterweights=np.array(filterweights), n_jobs=-1)#for eTRCA
        p_labels_withSAME_1,features_1 = estimator.fit(X_train_new, y_train_new).predict(test_data_MsCCA)  # for TRCA
        #estimator=FBMsCCA(filterbank=filterbank, n_components=1, filterweights=np.array(filterweights), n_jobs=-1)#for MsCCA
        #p_labels_withSAME,features = estimator.fit(X=np.squeeze(X_train_new),y=y_train_new, Yf=Yf).predict(test_data_MsCCA)# for MsCCA

        ####TDCA
        same.fit(data1c_train_TDCA , data1c_train_label)# for TDCA
        X_aug, y_aug = same.augment()
        print('original shape is:',data1c_train_.shape,'augmented shape is:',X_aug.shape)
        y_train_new = np.concatenate((data1c_train_label, y_aug), axis=0)
        X_train_new = np.concatenate((np.squeeze(data1c_train_TDCA), X_aug), axis=0) #for TDCA
        estimator = FBTDCA(filterbank, padding_len=5, n_components=8,filterweights=np.array(filterweights), n_jobs=-1)# for TDCA
        #test_data_MsCCA_new=np.zeros((test_data_MsCCA.shape[0],test_data_MsCCA.shape[1],test_data_MsCCA.shape[2]+6)) ## padding 0 for testing data, in fact unnecessary
        #test_data_MsCCA_new[:, :, :test_data_MsCCA.shape[2]] = test_data_MsCCA
        p_labels_withSAME_2,features_2 = estimator.fit(X_train_new,y_train_new, Yf=Yf).predict(test_data_MsCCA)# for TDCA

        accs_1=np.mean(p_labels_withSAME_1 == label)
        print('TRCA accuracy is:',accs_1)       
        itr_test_SAME = itr(config.num_class, accs_1, config.T / rfs + 0.5)
        acc_all_block_MsCCA_1.append(accs_1)
        itr_all_block_MsCCA_1.append(itr_test_SAME)    

        accs_2=np.mean(p_labels_withSAME_2 == label)
        print('TDCA accuracy is:',accs_2)
        itr_test_SAME = itr(config.num_class, accs_2, config.T / rfs + 0.5)
        acc_all_block_MsCCA_2.append(accs_2)
        itr_all_block_MsCCA_2.append(itr_test_SAME)               
        ############################Test#####################################
        inputs=test_data.to(device)
        inputs_cc=test_data_cc.to(device)
        labels=test_label.to(device)
        model.eval()
        if CNN=='MyNet':  
            outputs_label, outputs_label_1, outputs_label_m = model(inputs, inputs_cc)
            prediction = outputs_label + outputs_label_1 + outputs_label_m  
        elif CNN=='DNN':     
            outputs_label= model( inputs_cc)
            prediction = outputs_label 
        else:
            outputs_label = model(inputs)
            prediction = outputs_label 
        
        correct = (prediction.argmax(dim=1) == labels).sum().cpu().item()
        acc_test = correct / test_label.size(0)
        
        prediction = torch.nn.functional.normalize(prediction, dim=1)
        features_1=torch.from_numpy(features_1).to(device)
        features_1=torch.nn.functional.normalize(features_1, dim=1)

        features_2=torch.from_numpy(features_2).to(device)
        features_2=torch.nn.functional.normalize(features_2, dim=1)
        prediction_final=prediction+(features_1+features_2)/2
        correct_final = (prediction_final.argmax(dim=1) == test_label.to(device)).sum().cpu().item()
        acc_test_final = correct_final / test_label.size(0)
        
        itr_test = itr(config.num_class, np.mean(acc_test), config.T / rfs + 0.5)
        acc_all_block.append(acc_test)
        itr_all_block.append(itr_test)
        itr_test_final = itr(config.num_class, np.mean(acc_test_final), config.T / rfs + 0.5)
        acc_all_block_final.append(acc_test_final)
        itr_all_block_final.append(itr_test_final)
        print('-------------------------------------------------')
        print(name_test[0], 'cv...' + str(cv), " Test set results:", "Accuracy= {:.4f}".format(acc_test))
        print(name_test[0], 'cv...' + str(cv), " Final Test set results:", "Accuracy= {:.4f}".format(acc_test_final))

    acc_test = np.mean(acc_all_block)
    itr_test = np.mean(itr_all_block)
    PRmatrix[id_name] = acc_test
    PRmatrix_itr[id_name] = itr_test
    
    acc_test_MsCCA_1 = np.mean(acc_all_block_MsCCA_1)
    itr_test_MsCCA_1 = np.mean(itr_all_block_MsCCA_1)
    PRmatrix_MsCCA_1[id_name] = acc_test_MsCCA_1
    PRmatrix_itr_MsCCA_1[id_name] = itr_test_MsCCA_1

    acc_test_MsCCA_2 = np.mean(acc_all_block_MsCCA_2)
    itr_test_MsCCA_2 = np.mean(itr_all_block_MsCCA_2)
    PRmatrix_MsCCA_2[id_name] = acc_test_MsCCA_2
    PRmatrix_itr_MsCCA_2[id_name] = itr_test_MsCCA_2

    acc_test_final = np.mean(acc_all_block_final)
    itr_test_final = np.mean(itr_all_block_final)
    PRmatrix_final[id_name] = acc_test_final
    PRmatrix_itr_final[id_name] = itr_test_final

    
    if (dist.get_rank() == 0):
        num += 1
        print(name_test[0], " Test set results:", "Accuracy= {:.4f}".format(acc_test))
        print(name_test[0], " Test set itr:", "Itr= {:.4f}".format(itr_test))
        print('CNN accuracy is:',PRmatrix * 100)
        print(np.sum(PRmatrix * 100) / num)
        print('----------------------------')
        print(name_test[0], " TRCA Test set results:", "Accuracy= {:.4f}".format(acc_test_MsCCA_1))
        print(name_test[0], " TRCA Test set itr:", "Itr= {:.4f}".format(itr_test_MsCCA_1))
        print('TRCA accuracy is:',PRmatrix_MsCCA_1 * 100)
        print(np.sum(PRmatrix_MsCCA_1 * 100) / num)
        print('----------------------------')
        print(name_test[0], " TDCA Test set results:", "Accuracy= {:.4f}".format(acc_test_MsCCA_2))
        print(name_test[0], " TDCA Test set itr:", "Itr= {:.4f}".format(itr_test_MsCCA_2))
        print('TDCA accuracy is:',PRmatrix_MsCCA_2 * 100)
        print(np.sum(PRmatrix_MsCCA_2 * 100) / num)
        print('----------------------------')
        print(name_test[0], " final Test set results:", "Accuracy= {:.4f}".format(acc_test_final))
        print(name_test[0], " final Test set itr:", "Itr= {:.4f}".format(itr_test_final))
        print('final accuracy is:',PRmatrix_final * 100)
        print(np.sum(PRmatrix_final * 100) / num)
    dist.barrier()

if is_dataset == 0:
    name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
elif is_dataset == 1:
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35']
else:
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
            'S47', 'S48', 'S49', 'S50', 'S51',
            'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
            'S67', 'S68', 'S69', 'S70']
PRmatrix = PRmatrix * 100
acc = np.mean(PRmatrix)
var = np.var(PRmatrix)
std = np.sqrt(var)
std = std
itr = np.mean(PRmatrix_itr)
itr_var = np.var(PRmatrix_itr)
itr_std = np.sqrt(itr_var)


PRmatrix_MsCCA_1 = PRmatrix_MsCCA_1 * 100
acc_MsCCA_1 = np.mean(PRmatrix_MsCCA_1)
var_MsCCA_1 = np.var(PRmatrix_MsCCA_1)
std_MsCCA_1 = np.sqrt(var_MsCCA_1)
std_MsCCA_1 = std_MsCCA_1
itr_MsCCA_1 = np.mean(PRmatrix_itr_MsCCA_1)
itr_var_MsCCA_1 = np.var(PRmatrix_itr_MsCCA_1)
itr_std_MsCCA_1 = np.sqrt(itr_var_MsCCA_1)


PRmatrix_MsCCA_2 = PRmatrix_MsCCA_2 * 100
acc_MsCCA_2 = np.mean(PRmatrix_MsCCA_2)
var_MsCCA_2 = np.var(PRmatrix_MsCCA_2)
std_MsCCA_2 = np.sqrt(var_MsCCA_2)
std_MsCCA_2 = std_MsCCA_2
itr_MsCCA_2 = np.mean(PRmatrix_itr_MsCCA_2)
itr_var_MsCCA_2 = np.var(PRmatrix_itr_MsCCA_2)
itr_std_MsCCA_2 = np.sqrt(itr_var_MsCCA_2)

PRmatrix_final = PRmatrix_final  * 100
acc_final  = np.mean(PRmatrix_final )
var_final  = np.var(PRmatrix_final )
std_final  = np.sqrt(var_final )
std_final = std_final 
itr_final  = np.mean(PRmatrix_itr_final )
itr_var_final  = np.var(PRmatrix_itr_final )
itr_std_final  = np.sqrt(itr_var_final )

if (dist.get_rank() == 0):
    save_path = os.path.join(config.save_path, model_name, 'dataset_' + str(config.Dataset))
    log_write_test_final = open(os.path.join(save_path, CNN+'_stimulus_' + str(config.T / rfs) + 's_' + str(LOOC) + "_shot_test.txt"), "w")
    log_write_test_final.write('the acc  of proposed is:' + "\n" + str(PRmatrix) + "\n")
    log_write_test_final.write('the itr  of proposed is:' + "\n" + str(PRmatrix_itr) + "\n")
    log_write_test_final.write('the mean acc of proposed is:' + str(acc) + "+-" + str(std / np.sqrt(len(name))) + "\n")
    log_write_test_final.write('the mean itr of proposed is:' + str(itr) + "+-" + str(itr_std / np.sqrt(len(name))) + "\n")

    log_write_test_final.write('the acc  of TRCA is:' + "\n" + str(PRmatrix_MsCCA_1) + "\n")
    log_write_test_final.write('the itr  of TRCA is:' + "\n" + str(PRmatrix_itr_MsCCA_1) + "\n")
    log_write_test_final.write('the mean acc of TRCA is:' + str(acc_MsCCA_1) + "+-" + str(std_MsCCA_1 / np.sqrt(len(name))) + "\n")
    log_write_test_final.write('the mean itr of TRCA is:' + str(itr_MsCCA_1) + "+-" + str(itr_std_MsCCA_1 / np.sqrt(len(name))) + "\n")

    log_write_test_final.write('the acc  of TDCA is:' + "\n" + str(PRmatrix_MsCCA_2) + "\n")
    log_write_test_final.write('the itr  of TDCA is:' + "\n" + str(PRmatrix_itr_MsCCA_2) + "\n")
    log_write_test_final.write('the mean acc of TDCA is:' + str(acc_MsCCA_2) + "+-" + str(std_MsCCA_2 / np.sqrt(len(name))) + "\n")
    log_write_test_final.write('the mean itr of TDCA is:' + str(itr_MsCCA_2) + "+-" + str(itr_std_MsCCA_2 / np.sqrt(len(name))) + "\n")
    
    log_write_test_final.write('the acc  of final is:' + "\n" + str(PRmatrix_final) + "\n")
    log_write_test_final.write('the itr  of final is:' + "\n" + str(PRmatrix_itr_final) + "\n")
    log_write_test_final.write('the mean acc of final is:' + str(acc_final) + "+-" + str(std_final / np.sqrt(len(name))) + "\n")
    log_write_test_final.write('the mean itr of final is:' + str(itr_final) + "+-" + str(itr_std_final / np.sqrt(len(name))) + "\n")
# python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 main.py
