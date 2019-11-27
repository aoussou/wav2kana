#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 05:28:17 2019

@author: john
"""

import pandas as pd
import json
import numpy as np
import os
from pydub import AudioSegment

###############################################################################
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
###############################################################################  

csv_file_name = 'kore-words.csv'
word_audio_file_folder_name =  'kore-sound-vocab-munged'

csv_file_name = 'kore-sentences.csv'
word_audio_file_folder_name =  'kore-sound-sentences-munged'

root_dir = '../../data/audio/kore'

csv_file_path = os.path.join(root_dir,csv_file_name)
df = pd.read_csv(csv_file_path)
kana_list = df['kana']

correspondence_dict = json.load(open('./correspondence.json'))


word_audio_dir = os.path.join(root_dir,word_audio_file_folder_name)
word_audio_npy_dir = os.path.join(root_dir,'word_audio_npy') 
create_dir(word_audio_npy_dir)

targets_dir = os.path.join(root_dir,'targets')
create_dir(targets_dir)

names_list = df['file_name']

n_max_audio = 0
n_max_target = 0

for i,kana in enumerate(kana_list):
        
    chars = list(kana)
    name = names_list[i]
    
    audio_path = os.path.join(word_audio_dir,name) + '.mp3'
    audio = AudioSegment.from_file(audio_path, format='mp3')
    audio_npy = np.array(audio.get_array_of_samples())
    audio_npy_scaled = audio_npy/2**15
    
    # If the audio file is 16-bit, the range is -2^15 2^15
    np.save(os.path.join(word_audio_npy_dir,name),audio_npy_scaled)

    if len(audio_npy_scaled) > n_max_audio:
        n_max_audio = len(audio_npy_scaled)

    target = []
    for char in chars:
        target.append(correspondence_dict[char])
          
            
    target = np.array(target)
    np.save(os.path.join(targets_dir,name),target)

    if len(target) > n_max_target:
        n_max_target = len(target)
                
    print(i,kana,target)


         
print('The longest audio npy length is',n_max_audio)
print('The longest audio npy length is',n_max_target)