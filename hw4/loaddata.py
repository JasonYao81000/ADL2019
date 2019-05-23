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
    def __init__(self, root):
        'Initialization'
        list_IDs = []
        labels = []
        self.root = root
        self.transform = transforms.Compose([transforms.ToTensor()])
        with open(self.root + 'cartoon_attr.txt', encoding="utf-8") as f:
            lines = f.readlines()
        lines = lines[2:]
        for line in lines:
            list_IDs.append(line.split('png')[0]+'png')
            labels.append(line.rstrip('\n').split('png ')[1].split(' '))
        self.labels = np.array(labels).astype(int)
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img = Image.open(self.root + 'images/' + ID)
        img = self.transform(img)
        label = self.labels[index]

        return img, label