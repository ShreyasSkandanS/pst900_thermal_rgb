import sys
import glob
import os
import numpy as np
import cv2
import pdb

class DatasetUtilities:
    """
    Basic utilities for working with the PST900 Dataset
    """
    def __init__(self, datapath, split_type):
        """
        Initialize utilities
        """
        self.rgb_path = os.path.join(datapath, split_type, 'rgb')
        self.depth_path = os.path.join(datapath, split_type, 'depth')
        self.thermal_path = os.path.join(datapath, split_type, 'thermal')
        self.thermal_raw_path = os.path.join(datapath, split_type, 'thermal_raw')
        self.label_path = os.path.join(datapath, split_type, 'labels')

        self.rgb_image_list = []
        self.depth_image_list = []
        self.thermal_image_list = []
        self.thermal_raw_image_list = []
        self.label_image_list = []

        self.num_samples = 0
        self.DEPTH_SCALING = 2560
        self.pst_colormap = self.init_colormap()

        self.read_dataset()

    def read_dataset(self):
        """
        Read image lists from each folder and ensure all images are present
        """
        self.rgb_image_list = glob.glob(self.rgb_path + '/*.png')
        self.depth_image_list = glob.glob(self.depth_path + '/*.png')
        self.thermal_image_list = glob.glob(self.thermal_path + '/*.png')
        self.thermal_raw_image_list = glob.glob(self.thermal_raw_path + '/*.png')
        self.label_image_list = glob.glob(self.label_path + '/*.png')
        self.rgb_image_list.sort()
        self.depth_image_list.sort()
        self.thermal_image_list.sort()
        self.thermal_raw_image_list.sort()
        self.label_image_list.sort()
        assert (len(self.label_image_list) == len(self.rgb_image_list))
        assert (len(self.label_image_list) == len(self.depth_image_list))
        assert (len(self.label_image_list) == len(self.thermal_image_list))
        assert (len(self.label_image_list) == len(self.thermal_raw_image_list))
        self.num_samples = len(self.label_image_list)

    def get_sample(self, index):
        """
        Return an {RGB, D, T, T_raw, Label} pair given an index
        """
        print("==== Loading sample ====")
        # Load an RGB Image 
        rgb_image = cv2.imread(self.rgb_image_list[index])
        print("Loaded RGB image | H: {}, W: {}, C: {}, Type: {}".format(
            rgb_image.shape[0],
            rgb_image.shape[1],
            rgb_image.shape[2],
            rgb_image.dtype)
        )
        # Load a Depth Image
        depth_image = cv2.imread(self.depth_image_list[index], -1)
        depth_image = depth_image / self.DEPTH_SCALING
        assert len(depth_image.shape) == 2
        print("Loaded Depth image | H: {}, W: {}, C: {}, Type: {}".format(
            depth_image.shape[0],
            depth_image.shape[1],
            1,
            depth_image.dtype)
        )
        # Load an 8-bit Thermal Image
        thermal_image = cv2.imread(self.thermal_image_list[index], 0)
        assert len(thermal_image.shape) == 2
        print("Loaded Thermal image | H: {}, W: {}, C: {}, Type: {}".format(
            thermal_image.shape[0],
            thermal_image.shape[1],
            1,
            thermal_image.dtype)
        )
        # Load a raw 16-bit Thermal Image
        thermal_raw_image = cv2.imread(self.thermal_raw_image_list[index], -1)
        assert len(thermal_raw_image.shape) == 2
        print("Loaded Thermal Raw image | H: {}, W: {}, C: {}, Type: {}".format(
            thermal_raw_image.shape[0],
            thermal_raw_image.shape[1],
            1,
            thermal_raw_image.dtype)
        )
        # Load human annotated per-pixel label
        label_image = cv2.imread(self.label_image_list[index], -1)
        assert len(label_image.shape) == 2
        print("Loaded label image | H: {}, W: {}, C: {}, Type: {}".format(
            label_image.shape[0],
            label_image.shape[1],
            1,
            label_image.dtype)
        )
        return rgb_image, depth_image, thermal_image, thermal_raw_image, label_image

    def fill_thermal(self, thermal_image):
        """
        Example hole filling of thermal image
        """
        hole_mask = (thermal_image == 0).astype(np.uint8)
        filled_thermal = cv2.inpaint(
            thermal_image, 
            hole_mask, 
            10, 
            cv2.INPAINT_TELEA
        )
        return filled_thermal

    def init_colormap(self):
        """
        Initialize colormap for PST dataset
        """
        pst_colormap = [
            [0,0,0],
            [0,0,255],
            [0,255,0],
            [255,0,0],
            [255,255,255],
        ]
        return pst_colormap

    def visualize_label(self, label):
        """
        Apply colormap to label
        """
        mapped_label = np.array(self.pst_colormap).astype(np.uint8)[label]
        return mapped_label
        

def main():
    pst900_path = '/home/shreyas/Dropbox/RGBDT_Segmentation/PST900_RGBT_Dataset/'
    split_type = 'test'

    # Instantiate utilities
    utils = DatasetUtilities(pst900_path, split_type)

    # Example dataset sample loader
    rgb, depth, thermal, thermal_raw, label = utils.get_sample(140)

    cv2.imshow("RGB", rgb)
    cv2.imshow("Thermal", thermal)
    cv2.waitKey(0)

    # Example hole filling for Thermal image
    thermal_filled = utils.fill_thermal(thermal)

    cv2.imshow("Thermal_Filled", thermal_filled)
    cv2.waitKey(0)

    # Example colormapping for Label image
    colormapped_label = utils.visualize_label(label)

    cv2.imshow("Label", colormapped_label)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
