#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:34:20 2019

@author: yiransun
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable



class Encoder(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_size=40,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.1,
                 lamda=40):
        
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.output_size = output_size
        self.lamda = lamda

        # Layers.
        self.layers = nn.ModuleList([
                
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            
            # hidden
            nn.Conv1d(77,50,4),
            nn.MaxPool1d(2),
            nn.Conv1d(50,30,2),
            nn.MaxPool1d(2),
            nn.Conv1d(25,15,2),
            nn.MaxPool1d(2),
            nn.Conv1d(15,10,2),
            nn.Linear(10*2, 15),nn.ReLU(),
               
            # output
            nn.Linear(15, self.output_size),nn.Tanh(),
        ])

    @property
    def name(self):
        return (
            'MLP'
            '-lambda{lamda}'
            '-in{input_size}-out{output_size}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
        ).format(
            lamda=self.lamda,
            input_size=self.input_size,
            output_size=self.output_size,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

    def estimate_fisher(self, mode='decoder',dataset, sample_size, batch_size=32):
        
        # sample loglikelihoods from the dataset.
        data_loader = DataLoader(dataset, batch_size=16,shuffle=True)
        loglikelihoods = []
        if(mode=='decoder'):
            for x, y, _ in data_loader:
                x = x.view(batch_size, -1)
                x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
                y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
                loglikelihoods.append(
                    F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
                )
                if len(loglikelihoods) >= sample_size // batch_size:
                    break
        else:
            for x, _ ,y in data_loader:
                x = x.view(batch_size, -1)
                x = Variable(x)
                y = Variable(y)
                loglikelihoods.append(
                    F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
                )
                if len(loglikelihoods) >= sample_size // batch_size:
                    break
            
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean() for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    
    
    
    
    
    