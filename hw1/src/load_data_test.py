# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 01:05:18 2019

@author: hb250
"""

#from urllib2 import Request, urlopen
import json
from pandas.io.json import json_normalize
n_workers=4
file = 'D:/ADL1/data/valid.json'
with open(file) as valid_file:
    dict_valid = json.load(valid_file)

utterances = []
for sample in dict_valid:
    utterances += (
        [message['utterance']
         for message in sample['messages-so-far']]
        + [option['utterance']
           for option in sample['options-for-next']]
    )
utterances = list(set(utterances))
chunks = [
    ' '.join(utterances[i:i + len(utterances) // n_workers])
    for i in range(0, len(utterances), len(utterances) // n_workers)
]

words = []
filtrate = re.compile(u'[^\u0041-\u005A | \u0061-\u007A]')
for i in range(len(chunks)):
    filtered_str = filtrate.sub(r' ', chunks[i])#replace
    words.append(filtered_str)
for i in range(len(words)):
    querywords = words[i].split()
    words[i] = ' '.join(querywords)