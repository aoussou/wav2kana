#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:54:29 2019

@author: john
"""
import os
from utils import AudioDataset, PostProcess
import json
import random
import numpy as np

audio_dir = '/home/john/hdd/data/audio/kore/word_audio_npy/'
target_dir = '/home/john/hdd/data/audio/kore/targets/'

from glob import glob
audio_dir = '/home/john/hdd/data/audio/kore/word_audio_npy/'
glob_pattern = os.path.join(audio_dir, '*')
audio_list = sorted(glob(glob_pattern), key=os.path.getctime)

target_dir = '/home/john/hdd/data/audio/kore/targets/'
glob_pattern = os.path.join(target_dir, '*')
target_list = sorted(glob(glob_pattern), key=os.path.getctime)

n_dataset = len(audio_list)
train_proportion = .9
n_train = int(.9*n_dataset)
inds = np.arange(n_dataset)
np.random.shuffle(inds)
inds_train = inds[:n_train]
inds_val = inds[n_train:]

audio_list_train = np.array(audio_list)[inds_train].tolist()
audio_list_val = np.array(audio_list)[inds_val].tolist()

target_list_train = np.array(target_list)[inds_train].tolist()
target_list_val = np.array(target_list)[inds_val].tolist()

n_audio_max = 80000
n_target_max = 9

dataset_train = AudioDataset(audio_list_train,target_list_train,n_audio_max,n_target_max)
dataset_val = AudioDataset(audio_list_val,target_list_val,n_audio_max,n_target_max)


lookup_dict = json.load(open('./lookup.json'))

postprocessor = PostProcess(lookup_dict)



random_audio, random_target = dataset_train[random.randint(0,len(audio_list))]

random_target = random_target.numpy().astype('int')
print(postprocessor.target2kana(random_target))


from torch.utils.data import DataLoader

batch_size_train = 32
train_loader = DataLoader(dataset_train, batch_size=batch_size_train,shuffle=True)

n_epoch = 10


for data in train_loader :
    
    
 

