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
hira2kata = {}
for i,char in enumerate(hiragana_list) :
    
    hira2kata[char] = phonetic_list[i]

hira2kata['ー'] = 'ー'
katakana_list = list(katakana_str)
for i,char in enumerate(katakana_str) :
    
    hira2kata[char] = phonetic_list[i]



phonetic_numbering = {}

counter = 0
for  char in phonetic_list :

    if char not in phonetic_numbering:

        phonetic_numbering[char] = counter
        
        counter += 1
  
 
phonetic_numbering['ー'] = len(phonetic_numbering)
char_numbering = {}

for char in hira2kata:    
    char_numbering[char] = phonetic_numbering[hira2kata[char]] + 1


reverse_lookup = {}
for char in char_numbering:   
    
    nbr = char_numbering[char]
    if nbr not in reverse_lookup :
        reverse_lookup[nbr] = hira2kata[char] 


char_numbering[''] = 0
reverse_lookup[0] = ''

with open('./hira2kata.json', 'w+') as f:
    json.dump(hira2kata, f)
f.close()

with open('./correspondence.json', 'w+') as f:
    json.dump(char_numbering, f)
f.close()

with open('./lookup.json', 'w+') as f:
    json.dump(reverse_lookup, f)
f.close()
