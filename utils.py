#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:43:12 2019

@author: john
"""

from torch.utils.data import Dataset
import numpy as np

class AudioDataset(Dataset) :
    
    def __init__(self,audio_list,target_list) :
        
        self.audio_list = audio_list
        self.target_list = target_list
        
    
    def __len__(self) :
        
        return len(self.audio_list)
    
    def __getitem__(self,idx) :        
        
        audio_npy = np.load(self.audio_list[idx])
        target_npy = np.load(self.target_list[idx])
        
        return audio_npy, target_npy
        

class PostProcess() :
    
    def __init__(self,lookup_dict) :
        
        self.lookup_dict = lookup_dict
        
    def target2kana(self,target) :
        
        str_ = ''
        for nbr in target :   
            str_ += self.lookup_dict[str(nbr)]
        
        return str_