# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:05:05 2019

@author: hb2506
"""
import numpy as np
from torchvision import transforms
from torch.utils import data
from PIL import Image


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, test, root):
        'Initialization'
        list_IDs = []
        labels = []
        self.test = test
        self.root = root
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                             ])
        if self.test:
            with open(self.root, encoding="utf-8") as f:
                lines = f.readlines()
            lines = lines[2:]
            for line in lines:
                labels.append(line.rstrip('\n').split(' '))
            self.labels = np.array(labels).astype(int)
        else:
            with open(self.root + 'cartoon_attr.txt', encoding="utf-8") as f:
                lines = f.readlines()
            lines = lines[2:]
            for line in lines:
                list_IDs.append(line.split('png')[0]+'png')
                labels.append(line.rstrip('\n').split('png ')[1].split(' '))
        self.labels = np.array(labels).astype(int)
        self.hairs = self.labels[:, 0:6]
        self.eyes = self.labels[:, 6:10]
        self.faces = self.labels[:, 10:13]
        self.glasses = self.labels[:, 13:15]
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        label = self.labels[index]
        hair_id = np.argmax(self.hairs[index])
        eye_id = np.argmax(self.eyes[index])
        face_id = np.argmax(self.faces[index])
        glasses_id = np.argmax(self.glasses[index])
        if self.test:          
            return label, hair_id, eye_id, face_id, glasses_id
        else:
            # Select sample
            ID = self.list_IDs[index]
    
            # Load data and get label
            img = Image.open(self.root + 'images/' + ID)
            img = self.transform(img)
    
            return img, label, hair_id, eye_id, face_id, glasses_id