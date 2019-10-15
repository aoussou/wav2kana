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
root_dir = '../../data/audio/kore'
csv_file_name = 'Optimized Kore - Sheet1.csv'
csv_file_path = os.path.join(root_dir,csv_file_name)
df = pd.read_csv(csv_file_path)
kana_list = df['Vocab-kana']

correspondence_dict = json.load(open('./correspondence.json'))

word_audio_file_folder_name =  'kore-sound-vocab-munged'
word_audio_dir = os.path.join(root_dir,word_audio_file_folder_name)
word_audio_npy_dir = os.path.join(root_dir,'word_audio_npy') 
create_dir(word_audio_npy_dir)

targets_dir = os.path.join(root_dir,'targets')
create_dir(targets_dir)

names_list = df['Vocab-sound-local']

for i,kana in enumerate(kana_list):
    
    # Some audio files are missing, skip them.
    if type(names_list[i]) == str:
        chars = list(kana)
        name = names_list[i][7:-5]
        
        audio_path = os.path.join(word_audio_dir,name) + '.mp3'
        
        if os.stat(audio_path).st_size != 0 :
            audio = AudioSegment.from_file(audio_path, format='mp3')
            audio_npy = np.array(audio.get_array_of_samples())
            audio_npy_scaled = audio_npy/2**15
            
            # If the audio file is 16-bit, the range is -2^15 2^15
            np.save(os.path.join(word_audio_npy_dir,name),audio_npy_scaled)

            target = []
            for char in chars:
        
                if char == '・':                        
                        break
                else:
                    target.append(correspondence_dict[char])
                  
                    
            target = np.array(target)
            np.save(os.path.join(targets_dir,name),target)
        
            print(i,kana,target)
         
