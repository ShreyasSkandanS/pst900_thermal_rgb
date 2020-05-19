#-----------------------------------------------------------------------
# 	       Code to produce Segmentation Output in PyTorch
#             Originally written by Eduardo Romera (Sept 2017)
#	        https://github.com/Eromera/erfnet_pytorch
# 		Modified and Adapted by Shreyas Shivakumar
#                         and Neil Rodrigues
# -----------------------------------------------------------------------

# ==========================================================================
# ======================== LIBRARIES and PACKAGES ==========================
# ==========================================================================
# General utilities
import numpy as np
import os
import importlib
import pdb
import time
import glob
import sys
sys.path.append('../')
from argparse import ArgumentParser
# Image Processing
from PIL import Image
from skimage.color import label2rgb
from skimage.io import imsave, imread
# PyTorch
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
# Evaluation tools
# Taken from https://github.com/Eromera/erfnet_pytorch
from eval_iou import iouEval
# Dataset definition & Loader
from pst_data_definition import SemanticSegmentation

# ==========================================================================
# ====================== GLOBAL VAR and PATH LAYOUT ========================
# ==========================================================================
# Number of classes used during training
NUM_CLASSES = 5
# Resolution of input image
IMG_HEIGHT = 360
IMG_WIDTH = 640
# Directory of model files
ARTIFACT_DETECTION_DIR = "/data/pst900_thermal_rgb/pstnet/data/"
# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "data/PST900_RGBT_Dataset/test/"
# ==========================================================================


# ==========================================================================
# =============================== UTILITIES ================================
# ==========================================================================
class ToLabel:
    """
    Transformations on Label Image
    """
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

class ToThermal:
    """
    Transformations on Thermal Image
    """
    def __call__(self, image):
        import cv2
        image = np.array(image).astype(np.uint8)
        mask = (image == 0).astype(np.uint8)
        numpy_thermal = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
        numpy_thermal = numpy_thermal / 255.0
        return torch.from_numpy(numpy_thermal).float().unsqueeze(0)

class ImageTransform(object):
    """
    Transformations on inference (image, label) pair
    """
    def __init__(self, height=IMG_HEIGHT):
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
        input = ToTensor()(input)
        target = ToLabel()(target)
        return input, target

class ImageTransform2(object):
    """
    Transformation on inference (image, label) pair,
    slightly modified for some networks
    """
    def __init__(self, height=IMG_HEIGHT):
        self.height = height

    def __call__(self, input, target):
        input = np.array(input).astype(np.uint8)
        image = input[:,:,0:3]
        thermal = input[:,:,3]
        image = Image.fromarray(image)
        thermal = Image.fromarray(thermal)
        image = Resize(self.height, Image.BILINEAR)(image)
        target = Resize(self.height, Image.NEAREST)(target)
        thermal = Resize(self.height,Image.BILINEAR)(thermal)
        image = ToTensor()(image)
        thermal = ToThermal()(thermal)
        input = torch.cat([image,thermal],0)
        target = ToLabel()(target)
        return input, target


# ==========================================================================
# =============================== MAIN EVAL ================================
# ==========================================================================
def main(MODEL_NAME, NUM_CHANNELS, ARGS_LOAD_WEIGHTS, ARGS_LOAD_MODEL, ARGS_SAVE_DIR):
    """
    Main inference code
    """
    module = __import__(MODEL_NAME)
    Net = getattr(module, "Net")
    display_time = False

    print ("---------- DATA PATHS: ----------")
    print ("Model File: " + ARTIFACT_DETECTION_DIR + ARGS_LOAD_MODEL + MODEL_NAME)
    print ("Weight File: " + ARTIFACT_DETECTION_DIR + ARGS_LOAD_WEIGHTS)

    # Initialize model
    model = Net(NUM_CHANNELS, NUM_CLASSES)    
    if MODEL_NAME in ['mavnet','mavnet_rgb' ,'original_unet','original_unet_rgb','resunet','erfnet_rgb','erfnet','pstnet','pstnet_thermal']:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Load weights
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("[weight not copied for %s]"%(name)) 
                continue
            own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(ARTIFACT_DETECTION_DIR + ARGS_LOAD_WEIGHTS))
    print ("Model and weights loaded..")
    print ("---------------------------------")
    model.eval()

    if(not os.path.exists(ARGS_INFERENCE_DIR)):
        print ("Problem finding Inference Directory. Check path and try again.")
  
    # Setup image transforms 
    co_transform = ImageTransform(height=IMG_HEIGHT) if MODEL_NAME not in ['pstnet_thermal','erfnet'] else ImageTransform2(height=IMG_HEIGHT)
    
    # Initialize dataset and loader
    dataset = SemanticSegmentation(
        root = ARGS_INFERENCE_DIR, 
        co_transform = co_transform,
        NUM_CHANNELS=NUM_CHANNELS 
    )
    loader = DataLoader(
        dataset, 
        num_workers = 8, 
        batch_size = 1, 
        shuffle = False
    )

    # Initialize evaluation meters
    iouEvalVal = iouEval(NUM_CLASSES)
    inf_ctr = 0

    for step, (images,labels) in enumerate(loader):
        # Load {image, label} pair, enable GPU access
        images = images.cuda()
        labels = labels.cuda()
        # Setup as PyTorch variable
        inputs = Variable(images, requires_grad=False)
        targets = Variable(labels, requires_grad=False)
        
        # Setup clock
        inf_time_in = time.time()
        # Perform inference
        outputs = model(inputs)
        # Stop clock
        inf_time_out = time.time()
        
        # Add result to running evaluation
        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

        # Colormap output and save
        label = outputs[0].max(0)[1].byte().cpu().data
        label_color = label.unsqueeze(0)
        filenameSave = ARGS_SAVE_DIR + "/inference_" + str(inf_ctr).zfill(6) + ".png"
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        label_save = ToPILImage()(label_color)
        label_save.save(filenameSave)

        # Print inference time if required
        if display_time: 
            print("Val image: {} | Latency: {} ms".format(step,(inf_time_out - inf_time_in)*1000.0))

        inf_ctr += 1
    
    iouVal, iou_val_classes = iouEvalVal.getIoU()
    print("=============== " + MODEL_NAME +" VALIDATION ==============")
    print("[Validation] mIOU                : {}".format(iouVal))
    print("[Validation] Background          : {}".format(iou_val_classes[0]))
    print("[Validation] Fire extinguisher   : {}".format(iou_val_classes[1]))
    print("[Validation] Backpack            : {}".format(iou_val_classes[2]))
    print("[Validation] Hand drill          : {}".format(iou_val_classes[3]))
    print("[Validation] Rescue randy        : {}".format(iou_val_classes[4]))
    print("=============================================================")

def model_inference_loader(MODEL_NAME,NUM_CHANNELS):
    """
    Set up paths and model / weight directories
    """    
    ARGS_LOAD_WEIGHTS = "weights/" + MODEL_NAME + "/model_best.pth"
    ARGS_LOAD_MODEL = "architectures/" + MODEL_NAME + "/"
    ARGS_SAVE_DIR = ARTIFACT_DETECTION_DIR + "/inference/" + MODEL_NAME
    sys.path.append(ARTIFACT_DETECTION_DIR + ARGS_LOAD_MODEL)

    module = __import__(MODEL_NAME)
    Net = getattr(module, "Net")

    main(MODEL_NAME,NUM_CHANNELS, ARGS_LOAD_WEIGHTS, ARGS_LOAD_MODEL, ARGS_SAVE_DIR)

if __name__ == '__main__':
    MODEL_NAME = "pstnet_thermal"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)



