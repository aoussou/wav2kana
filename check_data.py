#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 07:44:07 2019

@author: john
"""
import os
import json
import random
import numpy as np
import scipy.io.wavfile as wavf
from pydub import AudioSegment

root_dir = '../../data/audio/kore/'

lookup_dict = json.load(open('./lookup.json'))

word_audio_file_folder_name =  'kore-sound-vocab-munged'
word_audio_dir = os.path.join(root_dir,word_audio_file_folder_name)
word_audio_npy_dir = os.path.join(root_dir,'word_audio_npy') 

targets_dir = os.path.join(root_dir,'targets')
list_ = os.listdir(word_audio_npy_dir)

name = random.choice(list_)[:-4]

audio_npy_scaled = np.load(os.path.join(word_audio_npy_dir,name) + '.npy')

# NOTE: it is very important to use np.int16, otherwise the audio is completely
# altered!
audio_npy_scaled_back = np.int16(audio_npy_scaled*2**15)

original = AudioSegment.from_mp3(os.path.join(word_audio_dir,name) + '.mp3')
Fs = original.frame_rate

wavf.write('test.mp3',Fs,audio_npy_scaled_back)

targets_dir = os.path.join(root_dir,'targets')

target = np.load(os.path.join(targets_dir,name) + '.npy')
str_ = ''

for nbr in target :    
    str_ += lookup_dict[str(nbr)]
    
print(str_)