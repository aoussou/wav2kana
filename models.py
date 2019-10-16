#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:45:35 2019

@author: john
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Letter() :
    
    def __init__(self,n_class,n_features=None,use_batchnorm = False) :
        
        self.n_class = n_class
        self.n_features = n_features
        
        if n_features is None :
            self.no_prepocessing_model()
        
    def no_prepocessing_model(self) :
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 250, 48, 160),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 48, 2),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000,self.n_class, 1),
        )
        
        
    def forward(self,x) :
        
       x = self.cnn(x)
       x = F.log_softmax(x, dim=1)
       
       return x