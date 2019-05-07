import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import init, Module
import torch.nn.functional as F


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)

def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)

def corr_loss(output,target,delta=0.00001):
    x=output
    y=target
    
    vx=x-torch.mean(x)
    vy=y-torch.mean(y)
    
    cost=torch.sum(vx * vy)/(torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))+delta)
    cost=cost.abs()
    
    return cost


def fbeta_bce_loss(y_pred, y_true, beta = 2):
    beta_sq = beta ** 2
    tp_loss = torch.sum(y_true * (1 - F.binary_cross_entropy(y_pred, y_true)))
    fp_loss = torch.sum((1 - y_true) * F.binary_cross_entropy(y_pred, y_true))

    return - torch.mean((1 + beta_sq) * tp_loss / ((beta_sq * torch.sum(y_true)) + tp_loss + fp_loss))

def confusion_matrix(predicted,y):
    from sklearn.metrics import confusion_matrix
    tn,fp,fn,tp=confusion_matrix(y,predicted).ravel()
    return (tn,fp,fn,tp)
