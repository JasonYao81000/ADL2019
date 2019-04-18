# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 02:33:31 2019

@author: hb2506
"""
#import csv

    
        
#    print('[-] Output saved to {}'.format(output_path))

#import re
#import torch
#from gensim.models import Word2Vec

word_dict = {}
line = list()
     
fp = open('./data/language_model/corpus_tokenized.txt', "r", encoding = "utf-8")
# 變數 lines 會儲存 filename.txt 的內容
lines = fp.readlines()
# close file
fp.close()
# print content
with open('./data/language_model/corpus_vocab.txt', mode='w', encoding="utf-8") as f:
    f.write('</S>'+'\n')
    f.write('<S>'+'\n')
    f.write('<UNK>'+'\n')
    for i in range(len(lines)):
        line = lines[i].split( )  
        for word in line:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
                f.write(word+'\n')
