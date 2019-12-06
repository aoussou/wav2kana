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
import librosa

class AudioDataset(Dataset) :
    
    def __init__(self,audio_list,target_list,n_audio_max,n_target_max,random_pad = True,change_speed=True) :
        
        self.audio_list = audio_list
        self.target_list = target_list
        self.n_audio_max = n_audio_max
        self.n_target_max = n_target_max
        self.random_pad = random_pad
        self.change_speed = change_speed
        
    
    def __len__(self) :
        
        return len(self.audio_list)
    
    def __getitem__(self,idx) :        
    
        audio_npy = np.load(self.audio_list[idx])
        
        if self.change_speed :
            
            # choose a random stretching factor
            # with the librosa function use, a factor of f means the
            # resulting audio will be f time faster (and as result,
            # the length of the audio is divided by f)
            
            # So f can be as large as we want but there is a lower limit that
            # depends on the length of the original audio and that of the
            # maximum audio length that he model can accommodate
            
            factor_min = len(audio_npy)/self.n_audio_max
            stretch_factor = np.random.uniform(factor_min,1.5)
            audio_npy = librosa.effects.time_stretch(audio_npy, stretch_factor)
            

        diff_pad = self.n_audio_max - len(audio_npy)    
        
        if self.random_pad :    
            random_int = random.randint(0,diff_pad)
            audio_npy = np.pad(audio_npy,(random_int,diff_pad-random_int),'constant')
        else :
            audio_npy = np.pad(audio_npy,(0,diff_pad),'constant')
        target_npy = np.load(self.target_list[idx])
        
        target_lengths = len(target_npy)
        target_padded = np.zeros((self.n_target_max))
        target_padded[:len(target_npy)] = target_npy    
        
        audio_tensor = torch.unsqueeze(torch.tensor(audio_npy,device=torch.device('cuda'),dtype=torch.float),0)
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