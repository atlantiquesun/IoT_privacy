#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:25:18 2019

@author: yiransun
"""

import torch
import torch.nn.functional as F

def corr_loss(output,target):
    x=output
    y=target
    
    vx=x-torch.mean(x)
    vy=y-torch.mean(y)
    
    cost=torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    cost=cost.abs()
    
    return cost
    
    