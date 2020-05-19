import os
import torch
import torch.utils.data as data

import pdb
import sys

sys.path.append("FastSCNN")
sys.path.append("../")

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils_fscnn.metric import SegmentationMetric
from utils_fscnn.visualize import get_color_pallete

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

class Evaluator(object):
    def __init__(self,NUM_CHANNELS):
        MODEL_NAME = "fast_scnn" if NUM_CHANNELS==4 else "fast_scnn_rgb"
        self.model_name = MODEL_NAME
        
        self.outdir = ARTIFACT_DETECTION_DIR + "/inference/" + self.model_name
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        if MODEL_NAME == "fast_scnn":
            input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406,0.4], [.229, .224, .225,0.4]),
            ])
        else:
            input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])


        val_dataset = get_segmentation_dataset(num_channels=NUM_CHANNELS,root=ARGS_INFERENCE_DIR, split='val', mode='testval', transform=input_transform)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        
        WEIGHTS_PATH = ARTIFACT_DETECTION_DIR+"/weights/fast_scnn_rgb" if NUM_CHANNELS==3 else ARTIFACT_DETECTION_DIR+"/weights/fast_scnn"
        print("Weights Path:", WEIGHTS_PATH)
        self.model = get_fast_scnn("fast_scnn",num_channels=NUM_CHANNELS, aux=False, pretrained=True, root=WEIGHTS_PATH).cuda()
        
        
        self.metric = SegmentationMetric(val_dataset.num_class)
    
    def eval(self):
        self.model.eval()
        avg_iou = 0
        AVG_IOU=0
        for i, (image, label) in enumerate(self.val_loader):
            image = image.cuda()

            outputs = self.model(image)
            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            self.metric.update(pred, label)
            pixAcc, mIoU, IoU = self.metric.get()

            avg_iou = avg_iou+mIoU
            AVG_IOU = AVG_IOU+IoU
            predict = pred.squeeze(0)

            mask = get_color_pallete(predict) 
            mask.save(os.path.join(self.outdir, 'seg_{}.png'.format(i)))
        avg_iou=avg_iou/i   
        AVG_IOU = AVG_IOU/i
        
        print("============"+ self.model_name + " VALIDATION ===============")
        print("[Validation] mIOU                : {}".format(avg_iou))
        print("[Validation] Background          : {}".format(AVG_IOU[0]))
        print("[Validation] Fire extinguisher   : {}".format(AVG_IOU[1]))
        print("[Validation] Backpack            : {}".format(AVG_IOU[2]))
        print("[Validation] Hand drill          : {}".format(AVG_IOU[3]))
        print("[Validation] Rescue randy        : {}".format(AVG_IOU[4]))
        print("=============================================================")



if __name__ == '__main__':
    NUM_CHANNELS=4
    evaluator = Evaluator(NUM_CHANNELS)
    print('Testing model: ', "fast_scnn")
    
    evaluator.eval()
