import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path, seed=9487):
        'Initialization'
        # Fix the random seeds
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.image_ids = []
        self.labels = []
        with open(path + '/cartoon_attr.txt', encoding='utf-8') as f:
            self.sample_len = int(f.readline().strip('\n'))
            self.attributes = f.readline().strip('\n').split(' ')
            for _ in range(self.sample_len):
                elements = f.readline().strip('\n').split(' ')
                self.image_ids.append(elements[0])
                self.labels.append(np.array(elements[1:]).astype(np.int))
        self.image_ids = np.array(self.image_ids)
        self.labels = np.array(self.labels)
        self.image_path = path + '/images/'

        self.transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.sample_len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample image and get label
        image_id = self.image_ids[index]
        label = self.labels[index]
        
        # Read image and convert to RGB, then apply the transform
        image = Image.open(self.image_path + image_id)
        image = self.transform(image)
        return image, label
