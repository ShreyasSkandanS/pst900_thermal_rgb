import numpy as np
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    path_string = '{basename}{extension}'.format(basename=basename, extension=extension)
    return os.path.join(root, path_string)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class SemanticSegmentation(Dataset):
    def __init__(self, root, co_transform=None,depth=None, thermal=None):
        self.images_root = os.path.join(root, 'rgb')
        self.labels_root = os.path.join(root, 'labels')
        self.depth_root = os.path.join(root, 'depth')
        self.thermal_root = os.path.join(root, 'thermal')

        self.filenames = [image_basename(f) 
                for f in os.listdir(self.labels_root)]

        self.filenames.sort()
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        with open(image_path(self.depth_root, filename, '.png'), 'rb') as f:
            depth = load_image(f).convert('I')
        with open(image_path(self.thermal_root, filename, '.png'), 'rb') as f:
            thermal = load_image(f).convert('I')

        if self.co_transform is not None:
            image, label,depth,thermal = self.co_transform(image, label, depth, thermal)

        return image, label, depth, thermal

    def __len__(self):
        return len(self.filenames)
