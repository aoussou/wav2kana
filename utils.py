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
import torchaudio
import torch.nn.functional as F
import os
from pathlib import Path

class AudioDataset(Dataset) :
    
    def __init__(self,audio_list,target_dir,n_audio_max,n_target_max,random_pad = False,change_speed=False,return_path=False) :
        
        self.audio_list = sorted(audio_list)
        self.n_audio_max = n_audio_max
        self.target_dir = target_dir
        self.n_target_max = n_target_max
        self.random_pad = random_pad
        self.change_speed = change_speed
        self.return_path = return_path

    def __len__(self) :
        
        return len(self.audio_list)
    
    def __getitem__(self,idx) :        
    
        audio_path = self.audio_list[idx]
        audio_tensor = torchaudio.load(audio_path, normalization=True)[0][0]
        diff_pad = self.n_audio_max - len(audio_tensor)    
        if self.random_pad :    
            random_int = random.randint(0,diff_pad)
            audio_tensor = F.pad(audio_tensor,(random_int,diff_pad-random_int),'constant',0)
        else :
            audio_tensor = F.pad(audio_tensor,(int(diff_pad/2),diff_pad - int(diff_pad/2)),'constant',0)

        fileID = Path(audio_path).stem
        target_path = os.path.join(self.target_dir,fileID + '.npy')
        target_npy = np.load(target_path)
        target_lengths = len(target_npy)
        target_padded = np.zeros((self.n_target_max))
        target_padded[:len(target_npy)] = target_npy    
        
        audio_tensor = torch.unsqueeze(audio_tensor,0).cuda()
        target_tensor = torch.tensor(target_padded,device=torch.device('cuda'),dtype=torch.long)
        target_lengths_tensor = torch.tensor(target_lengths,device=torch.device('cuda'),dtype=torch.long)

        if self.return_path :
            return [audio_tensor,target_tensor,target_lengths_tensor,audio_path]
        else :
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
    
    def levenshtein(self,s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)
    
        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
    
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
