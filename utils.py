#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:43:12 2019

@author: john
"""

import torch
from torch.utils.data import Dataset
import random
import numpy as np
import copy

class AudioDataset(Dataset) :
    
    def __init__(self,audio_list,target_list,n_audio_max,n_target_max,random_pad = True) :
        
        self.audio_list = audio_list
        self.target_list = target_list
        self.n_audio_max = n_audio_max
        self.n_target_max = n_target_max
        self.random_pad = random_pad
        
    
    def __len__(self) :
        
        return len(self.audio_list)
    
    def __getitem__(self,idx) :        
    
        audio_npy = np.load(self.audio_list[idx])

        diff_pad = self.n_audio_max - len(audio_npy)    
        
        if self.random_pad :    
            random_int = random.randint(0,diff_pad)
            audio_padded = np.pad(audio_npy,(random_int,diff_pad-random_int),'constant')
        else :
            audio_padded = np.pad(audio_npy,(0,diff_pad),'constant')
        target_npy = np.load(self.target_list[idx])
        
        target_lengths = len(target_npy)
        target_padded = np.zeros((self.n_target_max))
        target_padded[:len(target_npy)] = target_npy    
        
        audio_tensor = torch.unsqueeze(torch.tensor(audio_padded,device=torch.device('cuda'),dtype=torch.float),0)
        target_tensor = torch.tensor(target_padded,device=torch.device('cuda'),dtype=torch.long)
        target_lengths_tensor = torch.tensor(target_lengths,device=torch.device('cuda'),dtype=torch.long)
        
        return [audio_tensor,target_tensor,target_lengths_tensor]
        

class PostProcess() :
    
    def __init__(self,lookup_dict) :
        
        self.lookup_dict = lookup_dict
        
    def target2kana(self,target,refine=False) :
        
        str_ = ''
        for nbr in target :   
            str_ += self.lookup_dict[str(nbr)]
            
        if refine:            
            str_ = self.refine(str_)
        
        return str_
    
    def refine(self,raw_word) :
        
        current_char = None
        str_ = ''
        for char in raw_word :
            
            if char != current_char :
                str_+= char
                current_char = copy.copy(char)
    
        return str_