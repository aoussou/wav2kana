#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 05:28:17 2019

@author: john
"""

import pandas as pd
import json
import numpy as np

root_dir = '../../data/audio/kore/'
csv_file_path = root_dir + 'Optimized Kore - Sheet1.csv'

df = pd.read_csv(csv_file_path)

kana_list = df['Vocab-kana']

correspondence_dict = json.load(open('./correspondence.json'))

for kana in kana_list:
    
    chars = list(kana)
    
    code = []
    for char in chars:

        if char == '・':
                
                break
                print(kana)
                
        else:
            code.append(correspondence_dict[char])
          
            
    code = np.array(code)

    print(kana,code)
 
