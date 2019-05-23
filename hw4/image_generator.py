import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path, seed=9487, test=False):
        'Initialization'
        # Fix the random seeds
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.test = test

        self.labels = []
        if self.test:
            with open(path, encoding='utf-8') as f:
                self.sample_len = int(f.readline().strip('\n'))
                self.attributes = f.readline().strip('\n').split(' ')
                for _ in range(self.sample_len):
                    elements = f.readline().strip('\n').split(' ')
                    self.labels.append(np.array(elements).astype(np.int))
        else:
            self.image_ids = []
            with open(path + '/cartoon_attr.txt', encoding='utf-8') as f:
                self.sample_len = int(f.readline().strip('\n'))
                self.attributes = f.readline().strip('\n').split(' ')
                for _ in range(self.sample_len):
                    elements = f.readline().strip('\n').split(' ')
                    self.image_ids.append(elements[0])
                    self.labels.append(np.array(elements[1:]).astype(np.int))
            self.image_ids = np.array(self.image_ids)
            self.image_path = path + '/images/'
            self.transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.labels = np.array(self.labels)
        self.attr_hair = [x for x in self.attributes if 'hair' in x]
        self.attr_eye = [x for x in self.attributes if 'eye' in x]
        self.attr_face = [x for x in self.attributes if 'face' in x]
        self.attr_glasses = [x for x in self.attributes if 'glasses' in x]
        self.hairs = self.labels[:, 0:len(self.attr_hair)]
        self.eyes = self.labels[:, len(self.attr_hair):len(self.attr_hair) + len(self.attr_eye)]
        self.faces = self.labels[:, len(self.attr_hair) + len(self.attr_eye):len(self.attr_hair) + len(self.attr_eye) + len(self.attr_face)]
        self.glasses = self.labels[:, len(self.attr_hair) + len(self.attr_eye) + len(self.attr_face):]
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.sample_len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample label
        label = self.labels[index]
        hair_idx = np.argmax(self.hairs[index])
        eye_idx = np.argmax(self.eyes[index])
        face_idx = np.argmax(self.faces[index])
        glasses_idx = np.argmax(self.glasses[index])
        if self.test: 
            return label, hair_idx, eye_idx, face_idx, glasses_idx
        else:
            # Read image and convert to RGB, then apply the transform
            image_id = self.image_ids[index]
            image = Image.open(self.image_path + image_id)
            image = self.transform(image)
            return image, label, hair_idx, eye_idx, face_idx, glasses_idx
