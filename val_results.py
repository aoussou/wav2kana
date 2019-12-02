#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:52:40 2019

@author: john
"""

import torch
import os
import argparse
from utils import AudioDataset, PostProcess
import json


torch.set_default_tensor_type('torch.cuda.FloatTensor')

lookup_dict = json.load(open('./lookup.json'))

postprocessor = PostProcess(lookup_dict)

from torch.utils.data import DataLoader



        

def infer(model,val_loader) :

    for data in val_loader :
        
        audio = data[0]
        targets = data[1].cpu().numpy().astype('int')     
        output = model(audio)
        print(output)
        outmax = torch.argmax(output,dim=1).cpu().numpy()
        
        
        for i, vec in enumerate(outmax):
            
            #print(postprocessor.target2kana(targets[i]),postprocessor.target2kana(vec,refine = True))

            print(postprocessor.target2kana(vec,refine = False))            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--model_path', 
                    default = None, type = str)  
        
    args = parser.parse_args()        
    model_path = args.model_path

    info_dict = json.load(open(os.path.join(model_path,'info_dict.json')))

#    audio_list_train = info_dict['audio_list_train']
#    target_list_train = info_dict['target_list_train'] 
    
    audio_list_val = info_dict['audio_list_val'] 
    target_list_val = info_dict['target_list_val']     
    n_audio_max = info_dict['n_audio_max']     
    n_target_max = info_dict['n_target_max']  

    dataset_val = AudioDataset(audio_list_val,target_list_val,n_audio_max,n_target_max)
    val_loader = DataLoader(dataset_val, batch_size=8,shuffle=False)

    model = torch.load(os.path.join(model_path,'model.pt'))
    model = model.eval()
    
    infer(model,val_loader)

 

