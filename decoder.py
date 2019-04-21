#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:12:09 2019

@author: yiransun
"""

from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, input_size=10, output_size=1,
                 hidden_size=10,
                 hidden_layer_num=5,
                 hidden_dropout_prob=.5,
                 lamda=40):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        self.lamda = lamda
        
        # Layers.
        self.linear1 = nn.Linear(input_size, 20)
        self.relu1=nn.ReLU()       
        self.linear2 = nn.Linear(20, 10)
        self.relu2=nn.ReLU()
        self.linear3 = nn.Linear(10,output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        x=self.linear3(x)
        x=self.sigmoid(x)        
        return x

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda