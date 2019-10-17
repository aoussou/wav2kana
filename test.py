#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:52:40 2019

@author: john
"""

import torch
import os


###############################################################################
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
from models import Wav2Letter
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
n_val = len(dataset_val)

lookup_dict = json.load(open('./lookup.json'))

# If you look at the lookup dictionary, you will see that there are 78 characters
# In order to use the CTC loss in PyToch, we add to add 1
n_class = 79

postprocessor = PostProcess(lookup_dict)

criterion = torch.nn.CTCLoss()

random_audio, random_target, _ = dataset_train[random.randint(0,len(audio_list))]
random_target = random_target.cpu().numpy().astype('int')
print(postprocessor.target2kana(random_target))

from torch.utils.data import DataLoader

batch_size_train = 64
val_loader = DataLoader(dataset_val, batch_size=batch_size_train,shuffle=False)


n_epoch = 100

model = Wav2Letter(n_class)
model = model.cuda()
model.load_state_dict(torch.load(os.path.join('models','kore_word_state_dict.pt')))

#model = torch.load(os.path.join('models','kore_word_model.pt'))

#model.load_state_dict(torch.load(full_path))
model = model.eval()
        
for data in val_loader :
    
    audio = data[0]
    targets = data[1].cpu().numpy().astype('int')     
    output = model(audio)
    outmax = torch.argmax(output,dim=1).cpu().numpy()
    
    
    for i, vec in enumerate(outmax):
        
        
        print(postprocessor.target2kana(targets[i]),postprocessor.target2kana(vec))
        
        #STOP
#        outmax = torch.argmax(vec,dim=2)
#        
#        STOP
#        print(vec.shape)
#        
#        o = vec
#        print(vec)
#        
#        STOP
#
#    STOP        





        



 

