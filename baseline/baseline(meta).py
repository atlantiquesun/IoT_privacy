#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:09:35 2019

@author: yiransun
"""

#Meta Baseline

import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

Baseline = nn.Sequential(
        nn.Conv1d(15,50,4),Flatten(),
        nn.Linear(3700,200), nn.ReLU(),
        nn.Linear(200,100), nn.ReLU(),
        nn.Linear(100,50),nn.ReLU(),
        nn.Linear(50,1),nn.Sigmoid()
        )
Baseline=Baseline.double()

def Baseline_train(model,
          train_datasets,test_datasets,
          epochs=20,batch_size=32,consolidate=True,
          fisher_estimation_sample_size=64,
          lr=1e-2, weight_decay=1e-5,
          loss_log_interval=30,
          eval_log_interval=50,
          cuda=False):
    
    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    model.train()
    
    for epoch in range(1, epochs+1):
        
        data_loader=DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
        
        precision=0
        count=0
        
        for (x,y,_) in data_loader:
            count+=1
            
            #propagation
            optimizer.zero_grad()
            scores=model(x)
            #print(scores.shape, y.shape)
            loss=criterion(scores,y)
            loss.backward()
            optimizer.step()
            
            #calculating precision
            _,predicted=scores.max(1)
            precision += (predicted == y).sum().float()/len(x)
            
        print("epoch",epoch,":",precision/count)
        
