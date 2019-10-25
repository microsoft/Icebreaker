import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import itertools
import zipfile
import requests
import numpy as np
import pickle
import json
from sklearn.utils import shuffle
from sklearn import preprocessing
import math
import copy

def UCI_preprocessing(data,split_ratio=0.8,reduce_train=1.):
    train_data, test_data, _, _ = train_test_split(data, data, test_size=1.-split_ratio
                                                   )
    train_size=int(reduce_train*train_data.shape[0])
    train_data=train_data[0:train_size,:]
    return train_data,test_data



def load_UCI_data(csv_file,root_dir,flag_normalize=True,min_data=0.,normalization='True_Normal',flag_shuffle=True):
    path = root_dir + '/d' + csv_file + '.xls'
    Data = pd.read_excel(path)
    Data_mat = Data.as_matrix()
    if flag_normalize == True:
        if normalization == 'True_Normal':
            Data_std = preprocessing.scale(Data_mat)
            Data_std[Data_std == 0] = 0.01
            Data_mat = Data_std
        elif normalization == '0_1_Normal':
            Data_std = (Data_mat - Data_mat.min(axis=0)) / (
                    Data_mat.max(axis=0) - Data_mat.min(axis=0))
            Data_mat = Data_std + min_data

        else:
            raise NotImplementedError
    if flag_shuffle==True:
        Data_mat = shuffle(Data_mat)
    return  Data_mat

class base_UCI_Dataset_PABELGAM(Dataset):
    '''
    Most simple dataset by explicit giving train and test data
    '''
    def __init__(self,data,transform=None,flag_GPU=True):
        self.Data=data
        self.Data=torch.from_numpy(data).float().cuda()
        self.flag_GPU=flag_GPU
        self.transform=transform
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self, idx):
        sample_x=self.Data[idx,0:-1]
        sample_y=self.Data[idx,-1:]
        if self.transform and self.flag_GPU==True:
            sample_x=self.transform(sample_x)
            sample_x=sample_x.cuda()
            sample_y = self.transform(sample_y)
            sample_y = sample_y.cuda()
        elif self.transform and not self.flag_GPU:
            sample_x=self.transform(sample_x)
            sample_y=self.transform(sample_y)
        return sample_x,sample_y


class ToTensor(object):
    '''
    Convert numpy array to Tensor
    '''
    def __call__(self,sample):
        return torch.from_numpy(sample).float()

def Test_UCI(model,log_likelihood_func,W_sample,overall_test,sigma_out,split=3):
    overall_test=torch.from_numpy(overall_test).float().cuda()
    test_input=overall_test[:,0:-1]
    test_target=overall_test[:,-1:]
    test_size=test_input.shape[0]
    out_dim=test_target.shape[-1]
    batch_size = int(math.ceil(overall_test.shape[0] / split))
    pre_idx = 0
    total_se = 0
    total_MAE = 0
    total_NLL = 0
    model=copy.deepcopy(model)
    for counter in range(split+1):
        idx=min((counter+1)*batch_size,test_input.shape[0])
        if pre_idx==idx:
            break
        data_input=test_input[pre_idx:idx,:]
        data_target=test_target[pre_idx:idx,:]

        # completion
        pred_mean=model.completion(data_input,W_sample) # Nw x Nz x N x out
        pred_mean = torch.mean(torch.mean(pred_mean, dim=0), dim=0)
        pred_tot_ll=log_likelihood_func(data_input,data_target,W_sample,sigma_out)

        pred_test = torch.tensor(pred_mean.data) # N x out
        ae = torch.sum(torch.abs(pred_test - data_target))
        ae = torch.tensor(ae.data)
        se = torch.sum((pred_test - data_target) ** 2)
        se = torch.tensor(se.data)
        total_se += se
        total_MAE += ae
        total_NLL += -pred_tot_ll
        pre_idx = idx

    RMSE = torch.sqrt(1./(test_size*out_dim) * total_se)
    RMSE = torch.tensor(RMSE.data)
    MAE = torch.tensor((1./(test_size*out_dim) * total_MAE).data)
    NLL = torch.tensor((1./(test_size*out_dim) * total_NLL).data)
    return RMSE,MAE,NLL




