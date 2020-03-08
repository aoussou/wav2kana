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


#torch.set_default_tensor_type('torch.cuda.FloatTensor')

lookup_dict = json.load(open('./lookup.json'))

postprocessor = PostProcess(lookup_dict)

from torch.utils.data import DataLoader



def infer(model,val_loader) :

    total_lev_dist = 0
    n_val = len(val_loader.dataset)
    for data in val_loader :
        
        audio = data[0]
        targets = data[1].cpu().numpy().astype('int')     
        output = model(audio)
        outmax = torch.argmax(output,dim=1).cpu().numpy()
        
        for i, vec in enumerate(outmax):
            
            original = postprocessor.target2kana(targets[i]) 
            predicted = postprocessor.target2kana(vec,refine = True)
            lev_dist = postprocessor.levenshtein(original,predicted)
            total_lev_dist += lev_dist/len(original)
            
            print(original, predicted,lev_dist,round(lev_dist/len(original),2))
            
        av_lev = total_lev_dist/n_val
        
    print('average Levenshtein distance',av_lev)
        
        
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--model_path', 
                    default = None, type = str)  
    parser.add_argument('-n', '--gpu_number', 
                        default = 0, type = int)
        
    args = parser.parse_args()        
    model_path = args.model_path
    gpu_nbr = args.gpu_number
    
    torch.cuda.set_device(gpu_nbr)
    info_dict = json.load(open(os.path.join(model_path,'info_dict.json')))

#    audio_list_train = info_dict['audio_list_train']
#    target_list_train = info_dict['target_list_train'] 
    
    sets = info_dict['sets'] 
    list_dict = info_dict['list_dict']
    
    for set_ in sets :
        
        
        set_dict = list_dict[set_['path']]
        audio_list_val = set_dict['audio_list_val'] 
        target_dir = info_dict['target_dir']     
        n_audio_max = info_dict['n_audio_max']     
        n_target_max = info_dict['n_target_max']  

        if len(audio_list_val) != 0 :
            
            print(set_['path'])
            dataset_val = AudioDataset(audio_list_val,target_dir,n_audio_max,n_target_max,random_pad = False,change_speed=False)
            
            val_loader = DataLoader(dataset_val, batch_size=8,shuffle=False)
        
            model = torch.load(os.path.join(model_path,'model.pt'),map_location='cuda:' + str(gpu_nbr) )
            model.to(gpu_nbr)
            model = model.eval()
            
            infer(model,val_loader)
    
            print('='*20)
            print()
 

