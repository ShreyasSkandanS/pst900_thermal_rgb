###############################################################
# Main code for training ERFNet-based Semantic Segmentation CNN
#                       April 2018
#   Shreyas Skandan Shivakumar | University of Pennsylvania
#               Adapted from Eduardo Romera
###############################################################

import os
import random
import time
import numpy as np
import torch
import math
import pdb
import importlib

import sys
sys.path.append('../')

from skimage.color import label2rgb

from config.SS_config_train import *

from PIL import Image, ImageOps
from argparse import ArgumentParser
from shutil import copyfile

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid

from pst_data_definition import SemanticSegmentation
#from SS_data_definition import SemanticSegmentation
from utils.util_transform import ToLabel, Colorize, ToDepth, ToThermal
from utils.util_iouEVAL import iouEval, getColorEntry

from tensorboardX import SummaryWriter
writerTB = SummaryWriter(ARGS_SAVE_DIR + '/tensorboard')

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

class ImageTransform(object):
    def __init__(self, enc, height=IMG_HEIGHT, depth=False, thermal=False):
        self.enc = enc
        self.height = height
        self.depth_flag = depth
        self.thermal_flag = thermal

    def __call__(self, input, target, depth=None, thermal=None):
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)
        input = ToTensor()(input)
        target = ToLabel()(target)

        if self.depth_flag is True:
            depth = Resize(self.height, Image.NEAREST)(depth)
            depth = ToDepth()(depth)

        if self.thermal_flag is True:
            thermal = Resize(self.height, Image.BILINEAR)(thermal)
            thermal = ToThermal()(thermal)

        if (self.depth_flag is True) and (self.thermal_flag is False):
            return input, target, depth
        elif (self.depth_flag is True) and (self.thermal_flag is True):
            return input, target, depth, thermal
        else:
            return input, target

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        # 2019-07-31: Changed from NLLLoss2d to NLLLoss
        self.loss = torch.nn.NLLLoss(weight=weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), \
                         targets)

def visualizeClassPredictions(inputs, outputs, epoch, viz_n):
    pred_list = []
    for idx in range(0, viz_n):
        op = outputs[idx, :, :, :]
        label = op.max(0)[1].byte().cpu().data
        colormap = [
            [0,0,0],
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,255],
        ]
        rgb_label = np.array(colormap).astype(np.uint8)[label]
        rgb_label = np.transpose(rgb_label, (2,0,1))
        pred_list.append(torch.from_numpy(rgb_label))

        input_list = [inputs[i,0:3,:,:] for i in range(0,viz_n)]
        pred_backg = [outputs[i,0,:,:].unsqueeze(0) for i in range(0,viz_n)]
        pred_firex = [outputs[i,1,:,:].unsqueeze(0) for i in range(0,viz_n)]
        pred_bpack = [outputs[i,2,:,:].unsqueeze(0) for i in range(0,viz_n)]
        pred_drill = [outputs[i,3,:,:].unsqueeze(0) for i in range(0,viz_n)]
        pred_randy = [outputs[i,4,:,:].unsqueeze(0) for i in range(0,viz_n)]

        grid_vis = make_grid(pred_list, normalize=False, nrow=1)
        writerTB.add_image('images/val_pred', grid_vis, epoch)
        grid_img = make_grid(input_list, normalize=False, nrow=1)
        writerTB.add_image('images/val_images', grid_img.cpu(), epoch)
        grid_backg = make_grid(pred_backg, normalize=True, nrow=1)
        writerTB.add_image('images/val_backg', grid_backg.cpu(), epoch)
        grid_firex = make_grid(pred_firex, normalize=True, nrow=1)
        writerTB.add_image('images/val_firex', grid_firex.cpu(), epoch)
        grid_bpack = make_grid(pred_bpack, normalize=True, nrow=1)
        writerTB.add_image('images/val_bpack', grid_bpack.cpu(), epoch)
        grid_drill = make_grid(pred_drill, normalize=True, nrow=1)
        writerTB.add_image('images/val_drill', grid_drill.cpu(), epoch)
        grid_randy = make_grid(pred_randy, normalize=True, nrow=1)
        writerTB.add_image('images/val_randy', grid_randy.cpu(), epoch)

def train(model, enc=False):
    save_prefix = 'decoder'
    best_acc = 0

    weight = torch.ones(NUM_CLASSES)
    weight[0] = CLASS_0_WEIGHT
    weight[1] = CLASS_1_WEIGHT
    weight[2] = CLASS_2_WEIGHT
    weight[3] = CLASS_3_WEIGHT
    weight[4] = CLASS_4_WEIGHT

    print("Total number of classes is: {}".format(NUM_CLASSES))

    DEPTH_FLAG = True
    THERMAL_FLAG = True

    co_transform = ImageTransform(
        enc, 
        height=IMG_HEIGHT, 
        depth = DEPTH_FLAG,
        thermal = THERMAL_FLAG
    )
    co_transform_val = ImageTransform(
        enc, 
        height=IMG_HEIGHT, 
        depth = DEPTH_FLAG,
        thermal = THERMAL_FLAG,
    )

    #dataset_train = SemanticSegmentation(ARGS_TRAIN_DIR, co_transform)
    #dataset_val = SemanticSegmentation(ARGS_VAL_DIR, co_transform_val)

    dataset_train = SemanticSegmentation(
        root = ARGS_TRAIN_DIR, 
        co_transform = co_transform, 
        depth = DEPTH_FLAG,
        thermal = THERMAL_FLAG
    )
    dataset_val = SemanticSegmentation(
        root = ARGS_VAL_DIR, 
        co_transform = co_transform_val, 
        depth = DEPTH_FLAG,
        thermal = THERMAL_FLAG
    )

    loader = DataLoader(
        dataset_train,
        num_workers = ARGS_NUM_WORKERS,
        batch_size = ARGS_BATCH_SIZE,
        shuffle = True
    )
    loader_val = DataLoader(
        dataset_val, 
        num_workers = ARGS_NUM_WORKERS,
        batch_size = ARGS_BATCH_SIZE,
        shuffle = False
    )

    if ARGS_CUDA:
        weight = weight.cuda()

    criterion = CrossEntropyLoss2d(weight)
    savedir = ARGS_SAVE_DIR

    with open(savedir + "/train_log.txt", "w") as dec_train_log:
        dec_train_log.write(
            str("loss,iou,class_0,class_1,class_2,class_3_class_4,learning_rate\n")
        )
    with open(savedir + "/val_log.txt", "w") as dec_val_log:
        dec_val_log.write(
            str("loss,iou,class_0,class_1,class_2,class_3,class_4,learning_rate\n")
        )
    with open(savedir + "/model.txt", "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(
        model.parameters(),
        OPT_LEARNING_RATE_INIT,
        OPT_BETAS,
        eps = OPT_EPS_LOW,
        weight_decay = OPT_WEIGHT_DECAY
    )

    start_epoch = 1
    if ARGS_RESUME:
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), \
            "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    lambda1 = lambda epoch: pow((1-((epoch-1)/ARGS_NUM_EPOCHS)),0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(start_epoch, ARGS_NUM_EPOCHS + 1):
        print("--------------- [TRAINING] Epoch #", epoch, "---------------")
        scheduler.step(epoch)
        epoch_loss = []
        time_train = []
        doIouTrain = ARGS_IOU_TRAIN
        doIouVal =  ARGS_IOU_VAL

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("Learning rate: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        #for step, (images, labels) in enumerate(loader):
        #for step, (images, labels, depths) in enumerate(loader):
        for step, (images, labels, depths, thermals) in enumerate(loader):
            start_time = time.time()

            if ARGS_CUDA:
                images = images.cuda()
                labels = labels.cuda()
                # added this for RGBD
                depths = depths.cuda()
                # added this for RGBDT
                thermals = thermals.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            # added this for RGBD
            depths = Variable(depths)
            # added this for RGBT
            thermals = Variable(thermals)

            #import pdb
            #pdb.set_trace()
            # added this for RGBD
            #inputs = torch.cat((inputs, depths), 1)
            # added this for RGB-T
            inputs = torch.cat((inputs, thermals), 1)
            #pdb.set_trace()

            outputs = model(inputs)#, only_encode=False)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(
                    outputs.max(1)[1].unsqueeze(1).data,
                    targets.data
                )

            if ARGS_STEPS_LOSS > 0 and step % ARGS_STEPS_LOSS == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print("Loss: {average} (epoch: {epoch}, step: {step}) | "
                      "Avg time per image: {avgtime} s".format(
                         average=average, 
                         epoch=epoch, 
                         step=step, 
                         avgtime=(sum(time_train) / len(time_train) / ARGS_BATCH_SIZE)
                     )
                )

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print ("Average loss after epoch : {avgloss}".format(
            avgloss=average_epoch_loss_train)
        )

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("IoU on training data after EPOCH: ", iouStr, "%")

        writerTB.add_scalar('train/average_loss', average_epoch_loss_train, epoch)
        writerTB.add_scalar('train/average_iou', iouTrain, epoch)
        writerTB.add_scalar('train/bg_iou', iou_classes[0], epoch)
        writerTB.add_scalar('train/firex_iou', iou_classes[1], epoch)
        writerTB.add_scalar('train/bag_iou', iou_classes[2], epoch)
        writerTB.add_scalar('train/drill_iou', iou_classes[3], epoch)
        writerTB.add_scalar('train/randy_iou', iou_classes[4], epoch)
        writerTB.add_scalar('train/learning_rate', usedLr, epoch)

        with open(savedir + "/train_log.txt", "a") as dec_train_log:
            dec_train_log.write(
                "%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                    average_epoch_loss_train, 
                    iouTrain*100, 
                    iou_classes[0], # background
                    iou_classes[1], # firex
                    iou_classes[2], # backpack
                    iou_classes[3], # drill
                    iou_classes[4], # randy-boy
                    usedLr
                )
            )

        print("\n--------------- [VALIDATING] Epoch #", epoch, "---------------")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        #for step, (images, labels) in enumerate(loader_val):
        #for step, (images, labels, depths) in enumerate(loader_val):
        for step, (images, labels, depths, thermals) in enumerate(loader_val):
            start_time = time.time()
            if ARGS_CUDA:
                images = images.cuda()
                labels = labels.cuda()
                # added this for RGBD
                depths = depths.cuda()
                # added this for RGBDT
                thermals = thermals.cuda()

            inputs = Variable(images, requires_grad=False)
            targets = Variable(labels, requires_grad=False)
            # added this for RGBD
            depths = Variable(depths, requires_grad=False)
            # added this for RGBDT
            thermals = Variable(thermals, requires_grad=False)
            # added this for RGBD
            #inputs = torch.cat((inputs, depths), 1)
            inputs = torch.cat((inputs, thermals), 1)
            outputs = model(inputs, False)

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            if (doIouVal):
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if ARGS_STEPS_LOSS > 0 and step % ARGS_STEPS_LOSS == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print("Testing loss: {average} (epoch: {epoch}, step: {step}) |"
                      " Avg time per image: {avgstep} s".format(
                         average=average, 
                         epoch=epoch, 
                         step=step, 
                         avgstep=(sum(time_val) / len(time_val) / ARGS_BATCH_SIZE))
                )

            viz_n = min(3, ARGS_BATCH_SIZE)
            if ARGS_STEPS_LOSS > 0 and step % 5 == 0 and step < 35:
                visualizeClassPredictions(inputs, outputs, epoch, viz_n)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_val_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("IoU on test data after epoch: ", iouStr, "%")

        writerTB.add_scalar('val/average_loss', average_epoch_loss_val, epoch)
        writerTB.add_scalar('val/average_iou', iouVal, epoch)
        writerTB.add_scalar('val/bg_iou', iou_val_classes[0], epoch)
        writerTB.add_scalar('val/firex_iou', iou_val_classes[1], epoch)
        writerTB.add_scalar('val/bag_iou', iou_val_classes[2], epoch)
        writerTB.add_scalar('val/drill_iou', iou_val_classes[3], epoch)
        writerTB.add_scalar('val/randy_iou', iou_val_classes[4], epoch)
        writerTB.add_scalar('val/learning_rate', usedLr, epoch)
        with open(savedir + "/val_log.txt", "a") as dec_val_log:
            dec_val_log.write("%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                average_epoch_loss_val, 
                iouVal*100, 
                iou_val_classes[0], 
                iou_val_classes[1], 
                iou_val_classes[2], 
                iou_val_classes[3], 
                iou_val_classes[4], 
                usedLr)
            )

        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal

        if not os.path.exists(savedir + '/models'):
            os.makedirs(savedir + '/models')

        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        filenameCheckpoint = savedir + '/models/checkpoint.pth.tar'
        filenameBest = savedir + '/models/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        
        filename = savedir + "/models/model-{epoch}.pth".format(epoch=epoch)
        filenamebest = savedir + "/models/model_best.pth"

        #if ARGS_EPOCHS_SAVE > 0 and step > 0 and step % ARGS_EPOCHS_SAVE == 0:
        if epoch % ARGS_EPOCHS_SAVE == 0:
            torch.save(model.state_dict(), filename)
            print("Saving to: {filename} (epoch: {epoch})".format(filename=filename, epoch=epoch))
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print("Saving to: {filenamebest} (epoch: {epoch})".format(
                filenamebest=filenamebest, epoch=epoch)
            )
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
        print ('\n\n')
    return(model)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model with best IoU Score..")
        torch.save(state, filenameBest)


def save_session_files():
    save_dir = ARGS_SAVE_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_src = ARGS_REPO_DIR + 'train/SS_network_design.py'
    model_dst = save_dir + '/SS_network_design.py'
    copyfile(model_src, model_dst)    
    print('[init] Copied model file to save directory..')

    if not os.path.exists(save_dir + '/dataloader/'):
        os.makedirs(save_dir + '/dataloader/')
    data_def_src = ARGS_REPO_DIR + 'train/SS_data_definition.py'
    data_def_dst = save_dir + '/dataloader/SS_data_definition.py'
    copyfile(data_def_src, data_def_dst)
    print('[init] Copied data loader file to save directory..')

    if not os.path.exists(save_dir + '/config/'):
        os.makedirs(save_dir + '/config/')
    config_def_src = ARGS_REPO_DIR + 'config/SS_config_train.py'
    config_def_dst = save_dir + '/config/SS_config_train.py'
    copyfile(config_def_src, config_def_dst)
    print('[init] Copied config file to save directory..')

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        own_state[name].copy_(param)
    return model

def main():
    model_file = importlib.import_module(ARGS_MODEL)
    model = model_file.UNet(NUM_CLASSES)

    save_session_files()

    if ARGS_CUDA:
        model = torch.nn.DataParallel(model).cuda()

    if ARGS_STATE:
        model = load_my_state_dict(model, torch.load(ARGS_STATE))

    print("#################### FULL NETWORK TRAINING ####################")
    if (not ARGS_STATE):
        #pretrainedEnc = next(model.children()).encoder
        model = model_file.UNet(NUM_CLASSES)#, encoder=pretrainedEnc)
        if ARGS_CUDA:
            model = torch.nn.DataParallel(model).cuda()
    model = train(model, False)
    writerTB.close()
    print("###################### TRAINING FINISHED ######################")

if __name__ == '__main__':
    main()
