#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:54:29 2019

@author: john
"""
import os
from utils import AudioDataset, PostProcess
from torch.utils.data import DataLoader
import json
import numpy as np
from models import Wav2Letter
import torch
from glob import glob
import argparse
import sys
import time
import torch.nn as nn

###############################################################################
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
###############################################################################  

lookup_dict = json.load(open('./lookup.json'))
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_val_split(sets,n_audio_max,n_target_max) :

    audio_list_train = []
    audio_list_val = []

    target_list_train = []
    target_list_val = []
    
    for set_ in sets :
    
        audio_dir = os.path.join(set_['path'],'word_audio_npy')
        glob_pattern = os.path.join(audio_dir, '*')
        audio_list = sorted(glob(glob_pattern), key=os.path.getctime)
        
        target_dir = os.path.join(set_['path'],'targets')
        glob_pattern = os.path.join(target_dir, '*')
        target_list = sorted(glob(glob_pattern), key=os.path.getctime)
        
        n_dataset = len(audio_list)
        n_train = int(set_['train_ratio']*n_dataset)
        
        inds = np.arange(n_dataset)
        np.random.shuffle(inds)
        inds_train = inds[:n_train]
        inds_val = inds[n_train:]
        
        audio_list_train += np.array(audio_list)[inds_train].tolist()
        audio_list_val += np.array(audio_list)[inds_val].tolist()
        
        target_list_train += np.array(target_list)[inds_train].tolist()
        target_list_val += np.array(target_list)[inds_val].tolist()
    
    
    return audio_list_train, target_list_train, audio_list_val, target_list_val

#random_audio, random_target, _ = dataset_train[random.randint(0,len(audio_list))]
#random_target = random_target.cpu().numpy().astype('int')
#print(postprocessor.target2kana(random_target))

def train(model,optimizer,criterion,train_loader,val_loader,n_epoch,save_path):
    
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    total_val_loss_old = 1e16

    for e in range(n_epoch) :
        
        total_training_loss = 0   
        model.train()
        for data in train_loader :
            
            optimizer.zero_grad()   
            
            audio = data[0]
            targets = data[1]
            target_lengths = data[2]        
            current_batch_size = audio.size()[0]
            output = model(audio)
            
            # this basically a tensor vector of the length the size of the current
            # batch size, each entry being the length of the predictions (determined in the model)
            input_lengths = torch.full(size=(current_batch_size,), fill_value=output.size()[-1], dtype=torch.long)
            
            # loss = ctc_loss(input, target, input_lengths, target_lengths)
            loss = criterion(output.transpose(1, 2).transpose(0, 1),targets,input_lengths,target_lengths)        
            total_training_loss += float(loss.cpu())
                    
            loss.backward()
            optimizer.step()
    
        total_val_loss = 0  
        model.eval()      
        for data in val_loader :
            
            audio = data[0]
            targets = data[1]
            target_lengths = data[2]        
            current_batch_size = audio.size()[0]
            output = model(audio)        
    
            input_lengths = torch.full(size=(current_batch_size,), fill_value=output.size()[-1], dtype=torch.long)
            loss = criterion(output.transpose(1, 2).transpose(0, 1),targets,input_lengths,target_lengths)
    
            total_val_loss += float(loss.cpu())
            
            if total_val_loss < total_val_loss_old and save_path is not None:
                
                total_val_loss_old = total_val_loss
                
                torch.save(model.state_dict(), os.path.join(save_path,'state_dict.pt') )       
                torch.save(model, os.path.join(save_path,'model.pt'))
           
    
        print(e,total_training_loss/n_train,total_val_loss/n_val)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s1', '--dataset1', 
                        default = None, type = str)  
    parser.add_argument('-r1', '--train_ratio1', 
                        default = .8, type = float)  
    
    parser.add_argument('-s2', '--dataset2', 
                        default = None, type = str) 
    parser.add_argument('-r2', '--train_ratio2', 
                        default = .8, type = float)  
    
    parser.add_argument('-s3', '--dataset3', 
                        default = None, type = str)  
    parser.add_argument('-r3', '--train_ratio3', 
                        default = .8, type = float)  
    
    parser.add_argument('-a', '--longest_audio_npy', 
                        default = 80000, type = int)  
    parser.add_argument('-t', '--longest_target', 
                        default = 9, type = int)      
    
    parser.add_argument('-d', '--save_dir', 
                        default = 'auto', type = str)      

    parser.add_argument('-g', '--multi_gpu', 
                        default = False, type = bool)    

    parser.add_argument('-b', '--batch_size', 
                        default = 64, type = int)    
    
    args = parser.parse_args()        
    path_set1 = args.dataset1
    path_set2 = args.dataset2
    path_set3 = args.dataset3    
    n_audio_max = args.longest_audio_npy
    n_target_max = args.longest_target
    multi_gpu = args.multi_gpu
    batch_size = args.batch_size
    
    info_dict = {}
    info_dict['n_audio_max'] = n_audio_max
    info_dict['n_target_max'] = n_target_max
    
    sets = []
    
    if not os.path.isdir(path_set1) :
        print('set1 path invalid.')
        sys.exit()
        
    print('Set1',os.path.basename(path_set1),'training data ratio:',args.train_ratio1)
    set1 = {'path' : path_set1, 'train_ratio' :  args.train_ratio1}
    
    sets.append(set1)
    
    if path_set2 is not None :
        
        if not os.path.isdir(path_set2) :
            print('set2 path invalid.')
            sys.exit()
            
        print('Set2',os.path.basename(path_set2),'training data ratio:',args.train_ratio2)
        set2 = {'path' : path_set2, 'train_ratio' :  args.train_ratio2}
        sets.append(set2)
        
    if path_set3 is not None :
        
        if not os.path.isdir(path_set3) :
            print('set3 path invalid.')
            sys.exit()     
        
        print('Set3',os.path.basename(path_set3),'training data ratio:',args.train_ratio3)        
        set3 = {'path' : path_set3, 'train_ratio' :  args.train_ratio3}    
        sets.append(set3)
        
    save_dir = args.save_dir
    if save_dir == '0':
        save_path = None
        print('ATTENTION: model will not be saved!!!')
    elif save_dir == 'auto' :
        launch_time = time.strftime("%Y-%m-%d_%H%M")
        save_path = os.path.join('models',launch_time) 
        print('Model will be saved in',save_path)
    else :
        save_path = save_dir
        print('Model will be saved in',save_path)
        
    # If you look at the lookup dictionary, you will see that there are 78 characters
    # In order to use the CTC loss in PyToch, we need to add 1
    n_class = 79
    n_epoch = 100

    model = Wav2Letter(n_class)
    model = model.cuda()
    if multi_gpu:
        model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())
    
    #postprocessor = PostProcess(lookup_dict)
    
    criterion = torch.nn.CTCLoss()

    audio_list_train, target_list_train, audio_list_val, target_list_val = \
        train_val_split(sets,n_audio_max,n_target_max)

    dataset_train = AudioDataset(audio_list_train,target_list_train,n_audio_max,n_target_max)
    dataset_val = AudioDataset(audio_list_val,target_list_val,n_audio_max,n_target_max)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,shuffle=False)
    
    info_dict['audio_list_train'] = audio_list_train
    info_dict['target_list_train'] = target_list_train
    info_dict['audio_list_val'] = audio_list_val
    info_dict['target_list_val'] = target_list_val    
    info_dict['sets'] = sets
    
    create_dir(save_path)
    json.dump( info_dict, open(os.path.join(save_path,'info_dict.json'),'w+'))    
    
    train(model,optimizer,criterion,train_loader,val_loader,n_epoch,save_path)