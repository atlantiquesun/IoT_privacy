#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:35:42 2019

@author: yiransun
"""

import torch
from torch import nn
from torch import autograd

class Meta(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,x): 
        x=self.encoder(x)
        x=self.decoder(x)
        return x