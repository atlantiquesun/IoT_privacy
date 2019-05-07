#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 01:07:24 2019

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
from loss import corr_loss
from utils import xavier_initialize, Flatten

def prepare_datasets(train_ratio=0.8):
    from data_loader import Data
    dataset=Data()
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return (train_dataset, test_dataset)   

code_len=10
encoder=Encoder(input_size=15,output_size=code_len)
decoder=Decoder(input_size=code_len,output_size=1)
thief=Decoder(input_size=code_len,output_size=1)
meta=Meta(encoder,decoder)

meta=meta.double()
thief=thief.double()

#xavier_initialize(meta)
#xavier_initialize(thief)

def train(model,
          train_datasets,test_datasets,
          epochs=20,batch_size=16,consolidate=True,
          fisher_estimation_sample_size=32,
          lr=0.0001, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False):
    
    #prepare the loss and the optimizer
    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    #set the model's mode to training mode
    model.train()
    
    for epoch in range(1, epochs+1):
        
        data_loader=DataLoader(train_datasets, batch_size=batch_size, shuffle=True)        
        count=0
        precision=0
        cum_loss=0
        
        for (x,y,_) in data_loader:
            count+=1
            
            #propagation
            optimizer.zero_grad()
            scores=model(x)
            loss=criterion(scores,y)
            cum_loss += loss
            loss.backward()
            optimizer.step()
            
            #calculating precision
            predicted=scores.round()
            precision += (predicted == y).sum().float()/len(x)
        
        print('epoch', epoch, '   loss:',cum_loss.data/count, '   precision:',precision.data/count)
        
        if consolidate:
            model.encoder.consolidate(model.encoder.estimate_fisher(dataset=train_datasets, 
                                                            mode='public',sample_size=fisher_estimation_sample_size,batch_size=32))
            print("consolidated!")

def train_p(encoder, thief,
          train_datasets,test_datasets,
          epochs=20,batch_size=32,consolidate=True,
          fisher_estimation_sample_size=32,
          lr=0.0001, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False): 
        
    #prepare the loss and the optimizer
    criterion=nn.BCELoss()
    optimizer_e=optim.Adam(encoder.parameters(), lr=lr,weight_decay=weight_decay)
    optimizer_t=optim.Adam(thief.parameters(), lr=lr, weight_decay=weight_decay)
    
    #set the model's mode to training mode
    encoder.train()
    thief.train()
    
    for epoch in range(1, epochs+1):
        
        data_loader=DataLoader(train_datasets, batch_size=16,shuffle=True)
        count=0
        precision=0
        
        
        for (x,_,y) in data_loader:
            
            count+=1
            #with detect_anomaly(): (used when debugging)
            
            #clear stored gradients
            optimizer_e.zero_grad()
            optimizer_t.zero_grad()
            
            #forward
            intermediate=encoder(x)
            scores=thief(intermediate)
            
            #BCE loss, update the thief
            loss_t=criterion(scores,y)
            loss_t.backward(retain_graph=True)
            optimizer_t.step()
            optimizer_t.zero_grad()
            
            #corr_loss, update the encoder (adding ewc loss in the process)
            optimizer_e.zero_grad()
            ewc_loss=encoder.ewc_loss(cuda=cuda)
            #loss_e=corr_loss(scores,y)+ewc_loss.double()
            loss_e=(-1)*criterion(scores,y)+ewc_loss.double()
            loss_e.backward()
            optimizer_e.step()
            optimizer_e.zero_grad()
            optimizer_t.zero_grad()
            
            #calculating precision
            predicted=scores.round()
            precision += (predicted == y).sum().float()/len(x)
        
        print('epoch', epoch, '   precision:',precision.data/count)
            
        if consolidate:
            encoder.consolidate(encoder.estimate_fisher(train_datasets, mode='private',
                                                            sample_size=fisher_estimation_sample_size))
            
    data_loader=DataLoader(test_datasets,batch_size=256,shuffle=True)
    for(x,_,y) in data_loader:
        score=encoder(x)
        score=thief(score)
        predicted=score.round()
        print('precision: ',(predicted==y).sum().float()/len(x))


def test(model, test_datasets,mode='public'):    
    #show test results
    data_loader=DataLoader(test_datasets,batch_size=len(test_datasets),shuffle=True)
    if mode=='public':
        for(x,y,_) in data_loader:
            scores=model(x)
            predicted=scores.round()
            print('precision: ',(predicted==y).sum().float()/len(x))
    else:
         for(x,_,y) in data_loader:
            scores=model(x)
            predicted=scores.round()
            print('precision: ',(predicted==y).sum().float()/len(x))
        
        
        
        
        
        
        
