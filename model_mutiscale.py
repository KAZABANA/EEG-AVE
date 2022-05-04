# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:33:20 2021

@author: user
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Dict
class LabelSmoothingCrossEntropy(nn.Module):
    y_int = True
    def __init__(self, eps:float=0.1, reduction='mean'):
        super( LabelSmoothingCrossEntropy,self).__init__()
        self.eps,self.reduction = eps,reduction
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long().squeeze(1), reduction=self.reduction)

    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out):    return out.argmax(dim=-1)

class DE_mutiscale_classifier_esemble_twoclass(nn.Module):
    def __init__(self,channel,resolution):
         super(DE_mutiscale_classifier_esemble_twoclass,self).__init__()
         self.fc1=nn.Linear(resolution,4)
         self.fc2=nn.Linear(channel*4,64)
         self.fc3=nn.Linear(64,64)
         self.fc4=nn.Linear(64,2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x=F.elu(x)
         x=x.reshape((x.shape[0],x.shape[1]*4))
         x=self.fc2(x)
         x=F.elu(x)
         x=self.fc3(x)
#         x=self.dropout2(x)
         x=F.elu(x)
         feature=x
         x=self.fc4(x)
         x=self.dropout1(x)
         return x,feature
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
            {"params": self.fc3.parameters(), "lr_mult": 1},
            {"params": self.fc4.parameters(), "lr_mult": 1},
                  ]
         return params