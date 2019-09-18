# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, 
            noutput-ninput, 
            (3, 3), 
            stride=2, 
            padding=1, 
            bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat(
            [self.conv(input), self.pool(input)], 
            1
        )
        output = self.bn(output)
        return F.relu(output)
    

class bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(
            chann, 
            chann//4, 
            (1, 1), 
            stride=1, 
            padding=(0,0), 
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(chann//4, eps=1e-03)
        self.conv1x3_2 = nn.Conv2d(
            chann//4, 
            chann//4, 
            (1,3), 
            stride=1, 
            padding=(0,1*dilated), 
            bias=True,
            dilation = (1,dilated)
        )
        self.conv3x1_2 = nn.Conv2d(
            chann//4, 
            chann//4, 
            (3, 1), 
            stride=1, 
            padding=(1*dilated,0), 
            bias=True, 
            dilation = (dilated,1)
        )
        self.conv1x1_3 = nn.Conv2d(
            chann//4, 
            chann, 
            (1,1), 
            stride=1, 
            padding=(0,0), 
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv1x1_1(input)
        output = self.bn1(output)
        #output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv1x1_3(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(
            chann, 
            chann, 
            (3, 1), 
            stride=1, 
            padding=(1,0), 
            bias=True
        )
        self.conv1x3_1 = nn.Conv2d(
            chann, 
            chann, 
            (1,3), 
            stride=1, 
            padding=(0,1), 
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(
            chann, 
            chann, 
            (3, 1), 
            stride=1, 
            padding=(1*dilated,0), 
            bias=True, 
            dilation = (dilated,1)
        )
        self.conv1x3_2 = nn.Conv2d(
            chann, 
            chann, 
            (1,3), 
            stride=1, 
            padding=(0,1*dilated), 
            bias=True, 
            dilation = (1, dilated)
        )
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        model_path = "/data/Docker_Inference/RTFNET_erfnet/Unet_rtfnet_dataset/"
        weightpath = torch.load(model_path + 'models/model_best.pth')
        import sys
        sys.path.append(model_path)
        from nr_network_design import UNet as RGBNet
        rgb_model = RGBNet(9)
        rgb_model = torch.nn.DataParallel(rgb_model)
        rgb_model = rgb_model.cuda()
        rgb_model = self.load_my_state_dict(rgb_model, weightpath)
        for p in rgb_model.parameters():
            p.requires_grad = False        
        
        self.rgb_model = rgb_model 
        print('loaded weights!')
        #self.initial_block = DownsamplerBlock(3,16)
        # added this for RGBD (3 channel to 4 channel)
        self.initial_block = DownsamplerBlock(6,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 3):    #3 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        rgb_out = self.rgb_model(input[:,0:3,:,:])
        enc_in = torch.cat([rgb_out,input[:,3,:,:].unsqueeze(1)],dim=1)        
        output = self.initial_block(enc_in)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

    def load_my_state_dict(self, g_model, state_dict):
        own_state = g_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('not copying weights ')
                continue
            own_state[name].copy_(param)
        return g_model


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, 
            noutput, 
            4, 
            stride=2, 
            padding=1, 
            #output_padding=1, 
            bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        #print("Input shape: {}".format(input.shape))
        output = self.conv(input)
        #print("Output shape: {}".format(output.shape))
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, 
            num_classes, 
            2, 
            stride=2, 
            padding=0, 
            #output_padding=0, 
            bias=True
        )

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class Net(nn.Module):
    def __init__(self, num_classes, encoder=None): 
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    
            return self.decoder.forward(output)
