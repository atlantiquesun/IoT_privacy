#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:45:24 2019

@author: yiransun
"""
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import h5py
from random import randint, shuffle

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable, detect_anomaly
from torch.utils.data import DataLoader
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from meta import Meta
from utils import xavier_initialize, Flatten, corr_loss,confusion_matrix, fbeta_bce_loss

def prepare_datasets(train_ratio=0.8):
    from data_loader import Data
    dataset=Data()
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return (train_dataset, test_dataset)   

(train_dataset, test_dataset)=prepare_datasets()

code_len=10
prv=Encoder(input_size=15,output_size=code_len)
decoder=Decoder(input_size=code_len,output_size=1)
adv=Decoder(input_size=code_len,output_size=1)
meta=Meta(prv,decoder)

meta=meta.double()
adv=adv.double()

xavier_initialize(prv)
xavier_initialize(decoder)
xavier_initialize(adv)

def train(prv, adv, decoder,
          train_datasets,test_datasets,
          epochs=20,batch_size=32,consolidate=True,
          fisher_estimation_sample_size=32,
          lr=0.0001, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          k1=10, k2=10,# 1 batch for iteration
          cuda=False): 
        
    #prepare the loss and the optimizer
    criterion=nn.BCELoss()
    optimizer_e=optim.Adam(prv.parameters(), lr=lr, weight_decay=weight_decay) #for prv
    optimizer_t=optim.Adam(adv.parameters(), lr=lr, weight_decay=weight_decay) #for adv
    optimizer_d=optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay) #for decoder
    
    #set the model's mode to training mode
    prv.train()
    adv.train()
    decoder.train()
    
    for epoch in range(1, epochs+1):
        
        data_loader=DataLoader(train_datasets, batch_size=batch_size,shuffle=True)
        count=0
        precision_t=0
        precision_d=0
        
        count=0
        for (x,y,_) in data_loader:
            count+=1
            
            #propagation
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            intermediate=prv(x)
            scores=decoder(intermediate)
            loss=criterion(scores,y)
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()
            
            #calculating precision
            predicted=scores.round()
            precision_d += (predicted == y).sum().float()/len(x)
            
            if count>=k2:
                break
        
        print('                 decoder precision:',precision_d.data/count)
         
        count=0        
        for (x,_,y) in data_loader:
            
            count+=1
            #with detect_anomaly(): (used when debugging)
            
            #clear stored gradients
            optimizer_e.zero_grad()
            optimizer_t.zero_grad()
            
            #forward
            intermediate=prv(x)
            scores=adv(intermediate)
            
            #BCE loss, update the adv
            loss_t=criterion(scores,y)
            loss_t.backward(retain_graph=True)
            optimizer_t.step()
            
            #corr_loss, update the prv (adding ewc loss in the process)
            optimizer_e.zero_grad()
            loss_e=-criterion(scores,y)
            loss_e.backward()
            optimizer_e.step()
            
            #calculating precision
            predicted=scores.round()
            precision_t += (predicted == y).sum().float()/len(x)
            
            if count>=k1:
                break
        
        print('epoch', epoch, '   adversary precision:',precision_t.data/count)
    



def test(model, test_datasets,mode='public'):    
    #show test results
    data_loader=DataLoader(test_datasets,batch_size=len(test_datasets),shuffle=True)
    if mode=='public':
        for (x,y,_) in data_loader:
            scores=model(x)
            predicted=scores.round()
            print('precision: ',(predicted==y).sum().float()/len(x))
    else:
        for (x,_,y) in data_loader:
            scores=model(x)
            predicted=scores.round()
            print('precision: ',(predicted==y).sum().float()/len(x))
    
