import glob
import os
import PIL.Image
import numpy as np
import cv2
import pdb
import glob
import configparser
import numpy as np
import sys

class ClassWeights:
    """
    Calculate class weights for PST900
    """
    def __init__(self, datapath=''):
        """
        Initialize class
        """
        self.data_path = datapath
        self.label_path_train = os.path.join(datapath, 'train', 'labels')
        self.label_path_test = os.path.join(datapath, 'test', 'labels')
        self.label_stack = []
        self.label_paths = []
        self.num_classes = 5

    def process_labels(self):
        """
        Wrapper for processing all labels
        """
        train_labels = glob.glob(self.label_path_train + '/*.png')
        test_labels = glob.glob(self.label_path_test + '/*.png')
        self.label_paths = train_labels + test_labels
        print("Accumulating labels...")
        for label_img in self.label_paths:
            label = cv2.imread(label_img, -1)
            self.label_stack.append(label)
        print("Accumulating stack of labels done...")
        stack_np = np.stack(self.label_stack, axis=0)
        self.weights = self.calculate_class_weights(stack_np, self.num_classes)
        print("Weights are: {}".format(self.weights))

    def load_class_weights(self, weight_file):
        """ 
        Load class weights from .ini file 
        """
        config = configparser.ConfigParser()
        config.sections()
        config.read(weight_file)
        weights_mat = np.zeros([1, self.num_classes])
        weights_mat[0,0] = float(config['ClassWeights']['background'])
        weights_mat[0,1] = float(config['ClassWeights']['fire_extinguisher'])
        weights_mat[0,2] = float(config['ClassWeights']['backpack'])
        weights_mat[0,3] = float(config['ClassWeights']['drill'])
        weights_mat[0,4] = float(config['ClassWeights']['rescue_randy'])
        num_images = float(config['ClassWeights']['num_images'])
        print("Loaded class weights from .ini file...")
        return weights_mat.squeeze(), num_images

    def save_class_weights(self, weight_file):
        """
        Save class weights to .ini file
        """
        config = configparser.ConfigParser()
        config['ClassWeights'] = {}
        config['ClassWeights']['background'] = str(self.weights[0])
        config['ClassWeights']['fire_extinguisher'] = str(self.weights[1])
        config['ClassWeights']['backpack'] = str(self.weights[2])
        config['ClassWeights']['drill'] = str(self.weights[3])
        config['ClassWeights']['rescue_randy'] = str(self.weights[4])
        config['ClassWeights']['num_images'] = str(len(self.label_paths))
        with open(weight_file, 'w') as configfile:
            config.write(configfile)
        print("Saved class weights to .ini file...")

    def calculate_class_weights(self, Y, n_classes, method="paszke", c=1.02):
        """ Given the training data labels Calculates the class weights.
        Args:
           Y:      (numpy array) The training labels as class id integers.
                   The shape does not matter, as long as each element represents
                   a class id (ie, NOT one-hot-vectors).
           n_classes: (int) Number of possible classes.
           method: (str) The type of class weighting to use.
                   - "paszke" = use the method from from Paszke et al 2016
                               `1/ln(c + class_probability)`
           c:      (float) Coefficient to use, when using paszke method.
        Returns:
           weights:    (numpy array) Array of shape [n_classes] assigning a
                       weight value to each class.
        References:
           Paszke et al 2016: https://arxiv.org/abs/1606.02147
        """
        ids, counts = np.unique(Y, return_counts=True)
        n_pixels = Y.size
        p_class = np.zeros(n_classes)
        p_class[ids] = counts/n_pixels
        weights = 1/np.log(c+p_class)
        return weights

def main():

    pst900_path = '/home/shreyas/Dropbox/RGBDT_Segmentation/PST900_RGBT_Dataset/'

    weight_path = pst900_path + 'weights.ini'

    # Instantiate ClassWeights
    calc_weights = ClassWeights(pst900_path)

    # Example: to calculate weights for the entire dataset
    calc_weights.process_labels()

    # Example: to save weights to config file
    calc_weights.save_class_weights(weight_path)

    # Example: to load weights from config file
    weights, img_count = calc_weights.load_class_weights(weight_path)

if __name__ == '__main__':
    main()

