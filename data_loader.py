#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:25:05 2019

@author: yiransun
"""

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import h5py
from random import randint, shuffle

class Data(Dataset):
    def __init__(self,filename='/Users/yiransun/Desktop/OpportunityUCIDataset/opportunity.h5',
                 time_period=15,train_ratio=0.8):
        f=h5py.File(filename,'r')
        
        X=np.concatenate((np.asarray(f['train']['inputs']),np.asarray(f['test']['inputs'])))
        Y=np.concatenate((np.asarray(f['train']['targets']),np.asarray(f['test']['targets'])))
        
        n=X.shape[0]
        n_sensors=X.shape[1]
        q=n//time_period
        X=X[:q*time_period]
        X=X.reshape((q, time_period, n_sensors))
        #X=np.swapaxes(X, 2, 1) # shape = (n_samples, n_sensors || n_channels, n_w)
        
        y=np.zeros((q,))
        y_p=np.zeros((q,)) #the private hypothesis
        for i in range(q):
            if(np.any(Y[i*time_period:(i+1)*time_period]==6)):
                y[i]=1
            if(np.any(Y[i*time_period:(i+1)*time_period]==7)):
                y_p[i]=1
        
        
        inputs=[]
        for i in range(X.shape[0]):
            if y[i]==1 or y_p[i]==1:
                inputs.append([])
                inputs[-1].extend([X[i],y[i],y_p[i]])
            else:
                a=randint(1,36) #the proportion of data that score negative on either labels
                if(a==5):
                   inputs.append([])
                   inputs[-1].extend([X[i],y[i],y_p[i]]) 
        
        shuffle(inputs)
        targets_p=np.asarray([x[-1] for x in inputs])
        targets=np.asarray([x[-2] for x in inputs])
        targets=targets.reshape((targets.shape[0],1))
        targets_p=targets_p.reshape((targets_p.shape[0],1))
        inputs=np.asarray([x[0] for x in inputs])
        self.X=torch.from_numpy(inputs).double()
        self.Y=torch.from_numpy(targets).double()
        self.Y_p=torch.from_numpy(targets_p).double()
        
        '''
        n_train=round(len(inputs)*train_ratio)
        self.X_train=np.asarray(inputs[:n_train])
        self.y_train=np.asarray(targets[:n_train])
        self.y_p_train=np.asarray(targets_p[:n_train])
        
        self.X_test=np.asarray(inputs[n_train:])
        self.y_test=np.asarray(targets[n_train:])
        self.y_p_test=np.asarray(targets_p[n_train:])
    
        return (X_train,y_train,y_p_train,X_test,y_test,y_p_test)
        '''
    
    def __getitem__(self,index):
        return (self.X[index],self.Y[index],self.Y_p[index])
    
    def __len__(self):
        return self.X.shape[0] # of how many examples you have
    
dataset=Data()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


