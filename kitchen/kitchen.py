#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:44:14 2020

@author: john
"""

import torchaudio
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import torch.nn.functional as F


root_npy = '/home/john/ssd/audio/kore_words/kore-sound-vocab-munged/'
npys = sorted(os.listdir(root_npy))
int_ = random.randint(0,len(npys))
npy_path = root_npy + npys[int_]

root_mp3 = '/home/john/ssd/audio/kore_words/word_audio_npy/'
mp3s = sorted(os.listdir(root_mp3))
mp3_path = root_mp3 + mp3s[int_]

tens = torchaudio.load(npy_path,normalization=True)[0][0]
diff_pad = 80000 - len(tens)    
tens = F.pad(tens,(int(diff_pad/2),diff_pad - int(diff_pad/2)),'constant',0)

npy = np.load(mp3_path)
diff_pad = 80000 - len(npy)    
npy = np.pad(npy,(int(diff_pad/2),diff_pad - int(diff_pad/2)),'constant')


audio_npy0 = tens.numpy()
audio_npy = npy

diff = np.linalg.norm(audio_npy)  - np.linalg.norm(audio_npy0)
print('npy',audio_npy.shape,min(audio_npy),max(audio_npy))
print('tensor',audio_npy0.shape,min(audio_npy0),max(audio_npy0))
print(diff)



margin = 5000
tens = tens[margin:len(tens)-margin]
npy = npy[margin:len(npy)-margin]

from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(tens)
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(npy)