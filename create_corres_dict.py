#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 06:30:34 2019

@author: john
"""

import json

root_dir = './'

hiragana_str = 'ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをん'

katakana_str = 'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲン'
phonetic_kana_str = 'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチジッツズテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワオン'

hiragana_list = list(hiragana_str)
phonetic_list = list(phonetic_kana_str)
dict_ = {}
for i,char in enumerate(hiragana_list) :
    
    dict_[char] = phonetic_list[i]

katakana_list = list(katakana_str)
for i,char in enumerate(katakana_str) :
    
    dict_[char] = phonetic_list[i]

dict_['ー'] = 'ー'

phonetic_numbering = {}

for i, char in enumerate(phonetic_list) :
    
    phonetic_numbering[char] = i

dict_['ー'] = 'ー'    
phonetic_numbering['ー'] = len(phonetic_numbering)
char_numbering = {}

for char in dict_:
    
    char_numbering[char] = phonetic_numbering[dict_[char]]
    


with open(root_dir + 'correspondence.json', 'w+') as f:
    json.dump(char_numbering, f)
f.close()