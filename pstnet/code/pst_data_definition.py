import numpy as np
import os
import pdb

from PIL import Image, ImageCms

import torch
from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    path_string = '{basename}{extension}'.format(basename=basename, extension=extension)
    return os.path.join(root, path_string)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class SemanticSegmentation(Dataset):
    def __init__(self, root, co_transform=None,NUM_CHANNELS=4,depth=None,thermal=True):
        self.num_channels = NUM_CHANNELS
        self.images_root = os.path.join(root, 'rgb')
        self.labels_root = os.path.join(root, 'labels')
        self.depth = depth
        self.thermal = thermal
        if depth!=None: 
            self.depth_root = os.path.join(root,'depth')
        if thermal!=None: 
            self.thermal_root = os.path.join(root,'thermal')
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root)]
        self.filenames.sort()
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = Image.fromarray(np.array(load_image(f))).convert('RGB')

        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        if self.depth!=None:
            with open(image_path(self.depth_root, filename, '.png'), 'rb') as f:
                depth_image = load_image(f).convert('L')
        if self.thermal!=None:
            with open(image_path(self.thermal_root, filename, '.png'), 'rb') as f:
                thermal_image = load_image(f).convert('L')
        if self.co_transform is not None:
            image = Image.fromarray(np.dstack((np.array(image),np.array(thermal_image)))) if self.num_channels==4 else image
            image,label = self.co_transform(image,label)
        if self.thermal!=None:
            return image,label
        if self.depth!=None:
            return image,label,depth_image,thermal_image
        print(torch.max(image[3]),image.shape)
        return image, label

    def __len__(self):
        return len(self.filenames)
