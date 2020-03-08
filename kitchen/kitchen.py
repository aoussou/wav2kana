#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:44:14 2020

@author: john
"""

import torchaudio
import numpy as np
from matplotlib import pyplot as plt

tens = torchaudio.load('/home/john/ssd/audio/kore_words/kore-sound-vocab-munged/53e7857b4c062afc54e19da21dd2c280.mp3',normalization=True)[0][0].numpy()
npy = np.load('/home/john/ssd/audio/kore_words/word_audio_npy/53e7857b4c062afc54e19da21dd2c280.npy')
margin = 5000
tens = tens[margin:len(tens)-margin]
npy = npy[margin:len(npy)-margin]

from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(tens)
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(npy,linestyle=':')