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
    pdb.set_trace()
    return os.path.basename(os.path.splitext(filename)[0])

class SemanticSegmentation(Dataset):
    def __init__(self, root, co_transform=None, depth=None, thermal=None):
        self.images_root = os.path.join(root, 'rgb')
        self.labels_root = os.path.join(root, 'labels')
        self.depth = depth
        self.thermal = thermal

        if depth is True: 
            self.depth_root = os.path.join(root,'depth')

        if thermal is True: 
            self.thermal_root = os.path.join(root,'thermal')
        
        #pdb.set_trace()
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root)]

        self.filenames.sort()
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        # Load RGB
        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        # Load Label
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        # Load Depth
        if self.depth is True:
            with open(image_path(self.depth_root, filename, '.png'), 'rb') as f:
                #print("[debug] loaded depth image..")
                depth = load_image(f).convert('I')
        # Load Thermal
        if self.thermal is True:
            with open(image_path(self.thermal_root, filename, '.png'), 'rb') as f:
                #print("[debug] loaded thermal image..")
                thermal = load_image(f).convert('I')

        if self.co_transform is not None:
            if (self.depth is True) and (self.thermal is False):
                image,label,depth = self.co_transform(image, label, depth)
            elif (self.depth is True) and (self.thermal is True):
                image,label,depth,thermal = self.co_transform(image, label, depth, thermal)
            else:
                image,label = self.co_transform(image, label, depth, thermal)

        # Return RGB, Depth and Label
        if (self.depth is True) and (self.thermal is False):
            return image, label, depth
        # Retun RGB, Depth, Thermal and Label
        elif (self.depth is True) and (self.thermal is True):
            print(image.shape)
            return image, label, depth, thermal
        # Return RGB and Depth
        else:
            print(image.shape)
            return image, label

    def __len__(self):
        return len(self.filenames)
