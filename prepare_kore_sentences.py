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
kana_list = df['Sentence-kana']

correspondence_dict = json.load(open('./correspondence.json'))
hira2kata = json.load(open('./hira2kata.json'))


sentence_audio_file_folder_name =  'kore-sound-sentences-munged'
sentence_audio_dir = os.path.join(root_dir,sentence_audio_file_folder_name)
sentence_audio_npy_dir = os.path.join(root_dir,'sentence_audio_npy') 
create_dir(sentence_audio_npy_dir)

targets_dir = os.path.join(root_dir,'targets')
create_dir(targets_dir)

names_list = df['Sentence-sound-local']

n_max_audio = 0
n_max_target = 0

unnecessary_chars = ['<','>','b','/',' ','。']


with open("./katakana_sentences.txt","w+") as f:

    for i,kana in enumerate(kana_list):
        
        # Some audio files are missing, skip them.
        if type(names_list[i]) == str:
            chars = list(kana)
            
            str_ = ''
            
            for char in chars :
                
                if char not in unnecessary_chars :
                    
                    str_ += char
            
    
            name = names_list[i][7:-5]
            
            audio_path = os.path.join(sentence_audio_dir,name) + '.mp3'
            
            if os.stat(audio_path).st_size != 0 :
                audio = AudioSegment.from_file(audio_path, format='mp3')
                audio_npy = np.array(audio.get_array_of_samples())
                audio_npy_scaled = audio_npy/2**15
                
                # If the audio file is 16-bit, the range is -2^15 2^15
                np.save(os.path.join(sentence_audio_npy_dir,name),audio_npy_scaled)
    
                if len(audio_npy_scaled) > n_max_audio:
                    n_max_audio = len(audio_npy_scaled)
    
                target = []
                
                katakana_str = ''
                
                for char in str_:
                    
                    katakana_str += hira2kata[char]
            
                    if char == '・':                        
                            break
                    else:
                        target.append(correspondence_dict[char])

                f.write(katakana_str + '\n')


                target = np.array(target)
                np.save(os.path.join(targets_dir,name),target)
            
                if len(target) > n_max_target:
                    n_max_target = len(target)
                    
                
            
                print(i,kana,target)
   
f.close()      
print('The longest audio npy length is',n_max_audio)
print('The longest audio npy length is',n_max_target)