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
import argparse
###############################################################################
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
###############################################################################  

correspondence_dict = json.load(open('./correspondence_dict.json'))


def process(kana_list,audio_dir,targets_dir,audio_npy_dir) :   
    
    n_max_audio = 0
    n_max_target = 0

    for i,kana in enumerate(kana_list):
            
        chars = list(kana)
        name = names_list[i]
        
        ID = os.path.splitext(name)[0]
        
        audio_path = os.path.join(audio_dir,name)
        audio = AudioSegment.from_file(audio_path, format='mp3')
        audio_npy = np.array(audio.get_array_of_samples())
        audio_npy_scaled = audio_npy/2**15
        
        # If the audio file is 16-bit, the range is -2^15 2^15
        np.save(os.path.join(audio_npy_dir,ID),audio_npy_scaled)
    
        if len(audio_npy_scaled) > n_max_audio:
            n_max_audio = len(audio_npy_scaled)
    
        target = []
        for char in chars:
            target.append(correspondence_dict[char])
              
                
        target = np.array(target)
        np.save(os.path.join(targets_dir,ID),target)
    
        if len(target) > n_max_target:
            n_max_target = len(target)
                    
        print(i,kana,target)
                
    print('The longest audio npy length is', str(n_max_audio))
    print('The longest target length is', str(n_max_target))

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', 
                        default = '', type = str)  
    
    parser.add_argument('-r', '--ref_file', 
                        default = '', type = str) 
    
    parser.add_argument('-a', '--audio_folder', 
                        default = '', type = str)  
    
    args = parser.parse_args()
    
    data_dir = args.data_dir

    csv_file_name = args.ref_file
    dataset_name = os.path.splitext(csv_file_name)[0]
    
    audio_file_folder_name =  args.audio_folder   
    
    csv_file_path = os.path.join(data_dir,csv_file_name)
    df = pd.read_csv(csv_file_path)
    kana_list = df['katakana']
    
    audio_dir = os.path.join(data_dir,audio_file_folder_name)
    
    audio_npy_dir = os.path.join(data_dir,dataset_name,'word_audio_npy') 
    create_dir(audio_npy_dir)
    
    targets_dir = os.path.join(data_dir,dataset_name,'targets')
    create_dir(targets_dir)
    
    names_list = df['audio_file_name']    
    
    
    process(kana_list,audio_dir,targets_dir,audio_npy_dir)
    
    