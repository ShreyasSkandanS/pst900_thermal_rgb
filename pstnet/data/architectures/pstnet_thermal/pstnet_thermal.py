import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
sys.path.append('../data/')

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

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ninput, noutput, 3, 1, 1)
        )
        
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Net(nn.Module):
    def __init__(self,num_channels, num_classes): 
        super().__init__()

        rgb_net_weights = torch.load("../data/weights/pstnet/model_best.pth")
        from architectures.pstnet.pstnet import Net as RGBNet
        rgb_net = RGBNet(3,5)
        rgb_net = torch.nn.DataParallel(rgb_net)
        rgb_net = rgb_net.cuda()
        rgb_net = self.load_my_state_dict(rgb_net, rgb_net_weights)
        for p in rgb_net.parameters():
            p.requires_grad = False       

        self.rgb_net = rgb_net

        # ================== ENCODER ==================
        self.initial_block_enc = DownsamplerBlock(9, 32)

        self.layers_enc = nn.ModuleList()
        self.layers_enc.append(DownsamplerBlock(32,64))

        # Encoder 5 Stack
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 
        
        self.layers_enc.append(DownsamplerBlock(64,128))
        # Encoder 2 Stack
        # 1)
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 16))
        # 2)
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 16))

        # ================== DECODER ==================
        self.layers_dec = nn.ModuleList()
        self.layers_dec.append(UpsamplerBlock(128,64))
        self.layers_dec.append(non_bottleneck_1d(64, 0, 1))
        self.layers_dec.append(non_bottleneck_1d(64, 0, 1))

        self.layers_dec.append(UpsamplerBlock(64,16))
        self.layers_dec.append(non_bottleneck_1d(16, 0, 1))
        self.layers_dec.append(non_bottleneck_1d(16, 0, 1))

        # Final output
        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, num_classes, 3, 1, 1)
        )
        
    def forward(self, input):

        rgb_out = self.rgb_net(input[:,0:3,...])

        for bidx in range(0, rgb_out.shape[0]):
            tensor_min = torch.min(rgb_out[bidx, ...])
            tensor_max = torch.max(rgb_out[bidx, ...])
            tensor_range = tensor_max - tensor_min
            rgb_out[bidx, ...] = (rgb_out[bidx, ...] - tensor_min) / tensor_range

        input = torch.cat([rgb_out, input[:,3,...].unsqueeze(1), input[:,0:3,...]], 1)
        #import pdb; pdb.set_trace()

        output = self.initial_block_enc(input)
        for layer_enc in self.layers_enc:
            output = layer_enc(output)
        for layer_dec in self.layers_dec:
            output = layer_dec(output)
        output = self.output_conv(output)
        return output


    def load_my_state_dict(self, g_model, state_dict):
        own_state = g_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('[ERROR] Could not load weights correctly!')
                continue
            own_state[name].copy_(param)
        return g_model
