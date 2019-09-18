import numpy as np
import os
import pdb
from torch.utils.data import Dataset
from PIL import Image

def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    path_string = '{basename}{extension}'.format(basename=basename, extension=extension)
    return os.path.join(root, path_string)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class SemanticSegmentationInference(Dataset):

    def __init__(self, root, co_transform=None):
        self.images = root
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images)]
        self.filenames.sort()
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        if self.co_transform is not None:
            image = self.co_transform(image)
        return image

    def __len__(self):
        return len(self.filenames)
