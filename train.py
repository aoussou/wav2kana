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

audio_dir = '/home/john/hdd/data/audio/kore/word_audio_npy/'
target_dir = '/home/john/hdd/data/audio/kore/targets/'

from glob import glob
audio_dir = '/home/john/hdd/data/audio/kore/word_audio_npy/'
glob_pattern = os.path.join(audio_dir, '*')
audio_list = sorted(glob(glob_pattern), key=os.path.getctime)

target_dir = '/home/john/hdd/data/audio/kore/targets/'
glob_pattern = os.path.join(target_dir, '*')
target_list = sorted(glob(glob_pattern), key=os.path.getctime)


dataset = AudioDataset(audio_list,target_list)

lookup_dict = json.load(open('./lookup.json'))

postprocessor = PostProcess(lookup_dict)

random_audio, random_target = dataset[random.randint(0,len(audio_list))]

print(postprocessor.target2kana(random_target))