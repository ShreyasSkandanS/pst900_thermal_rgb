import torch
import torch.nn as nn
import torch.nn as nn 
import torch.nn.functional as F 
import pdb 
import math
USE_BIAS  = True 

choose_bn = {"normal": nn.BatchNorm2d}

class Net(nn.Module):
    """"
	Build a MAVNet  
    Args:
        X (4-D Tensor): (N, H, W, C)
        is_training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    """
    def __init__(
        self, in_channels=4, 
        n_classes=21, 
        is_deconv=False, 
        is_batchnorm=True, 
        keep_prob=1.0,
        n_downsampling=2,
        n_bn_blocks=4
    ):
        super(Net, self).__init__()
        self.is_deconv      = is_deconv
        self.in_channels    = in_channels
        self.is_batchnorm   = is_batchnorm
        self.feature_scale  = 1 
        self.n_downsampling = n_downsampling
        self.n_bn_blocks    = n_bn_blocks 

        # Number of features for each layer 
        self.first_layer_features = 4 
        filters = [self.first_layer_features] 
        for i in range(n_downsampling-1):
            filters.append(filters[-1]*2) # x2 features after each downsampling 
        for i in range(n_bn_blocks):
            filters.append(filters[-1])
        for i in range(n_downsampling-1):
            filters.append(filters[-1]//2) # /2 features after upsampling  
            filters.append(filters[-1])    # x1 features after nonbottleneck block 
        filters.append(filters[-1])        # x1 features after final upampling
        filters = [int(x * self.feature_scale) for x in filters]
        
        # downsampling
        self.down_layers = nn.ModuleList() 
        in_channels      = self.in_channels 
        for i in range(n_downsampling):
            out_channels = filters[i] 
            pool   = conv_conv_pool(in_channels,   out_channels, is_pool=True, keep_prob=keep_prob)
            in_channels = out_channels  
            self.down_layers.append(pool) 

        self.bn_layers  = nn.ModuleList()
        for i in range(n_bn_blocks):
            out_channels = filters[i+n_downsampling] 
            bn_layer    = DWFab_block(in_channels, out_channels, kernel_size=3, dilation_rate=2**(i+1)) 
            in_channels = out_channels  
            self.bn_layers.append(bn_layer) 

        # Upsampling
        self.up_layers = nn.ModuleList() 
        for i in range(n_downsampling - 1):
            out_channels = filters[i+n_downsampling+n_bn_blocks] 
            up1    = up2D(in_channels,             out_channels, is_deconv=is_deconv, scale_factor=2)
            up2    = non_bottleneck1D(out_channels,   out_channels, kernel_size=3, dilation_rate=1)
            in_channels = out_channels  
            self.up_layers.append(up1)
            self.up_layers.append(up2)

        self.outFn = nn.Sequential(
                     up2D(in_channels,             filters[-1], is_deconv=is_deconv, scale_factor=2),
                     nn.Conv2d(filters[-1],        n_classes, 1, 1, 0),
                     nn.ReLU())
        self.init_list = [self.down_layers, self.bn_layers, self.up_layers, self.outFn]
        self.__init_weights__()

    def __init_weights__(self):
        def _init_weights(m):
            # Initialize filters with Gaussian random weights                                     
            if isinstance(m, nn.Conv2d):                                    
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels    
                m.weight.data.normal_(0, math.sqrt(2. / n))                 
                if m.bias is not None:                                      
                    m.bias.data.zero_()                                     
            elif isinstance(m, nn.ConvTranspose2d):                         
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels     
                m.weight.data.normal_(0, math.sqrt(2. / n))                 
                if m.bias is not None:                                      
                    m.bias.data.zero_()                                     
            elif isinstance(m, nn.BatchNorm2d):                             
                m.weight.data.fill_(1)                                      
                m.bias.data.zero_()  

        for layer in self.init_list:
            layer.apply(_init_weights) 


    def forward(self, inputs):
        outputs     = inputs 
        #skips_list  = [] 
        for i in range(self.n_downsampling):
            outputs = self.down_layers[i](outputs)
            #skips_list.append(outputs)

        for i in range(self.n_bn_blocks):
            outputs = self.bn_layers[i](outputs)

        # Skip connection
        #outputs = skip + outputs 
        for i in range(len(self.up_layers)):
            #if len(self.up_layers) > self.n_downsampling-1: 
            	#outputs = outputs + skips_list[self.n_downsampling - int(i//2) - 1] 
            #else: 
            	#outputs = outputs + skips_list[self.n_downsampling - i - 1] 
            outputs = self.up_layers[i](outputs)

        #outputs     = skips_list[0] + outputs 
        outputs     = self.outFn(outputs) 

        return outputs  

class conv_conv_pool(nn.Module):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, C, H, W)
        n_filters (list): number of filters [int, int]
        is_training (1-D Tensor): Boolean Tensor
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    def __init__(
            self, 
            in_channels, 
            n_filters,
            k_size=3, 
            is_pool=True,
            is_batchnorm=True,
            batchnorm_type="normal",
            keep_prob=1.0):
        super(conv_conv_pool, self).__init__()
        self.keep_prob = keep_prob
        conv1 = nn.Conv2d(in_channels,
            n_filters,
            kernel_size=k_size,
            stride=1,
            padding=(k_size-1)//2,
			bias=USE_BIAS)

        conv2 = nn.Conv2d(n_filters,
            n_filters,
            kernel_size=k_size,
            stride=1,
            padding=(k_size-1)//2,
			bias=USE_BIAS)

        
        batchnorm_fn = choose_bn[batchnorm_type]       
        if is_batchnorm:
            self.conv_block1 = nn.Sequential(
                    conv1, 
                    batchnorm_fn(n_filters), 
                    nn.ReLU()) 
            self.conv_block2 = nn.Sequential(
                    conv2, 
                    batchnorm_fn(n_filters), 
                    nn.ReLU())
        else:
            self.conv_block1 = nn.Sequential(
                    conv1, 
                    nn.ReLU()) 
            self.conv_block2 = nn.Sequential(
                    conv2, 
                    nn.ReLU())
        if is_pool:
            self.conv_block2 = nn.Sequential(self.conv_block2, nn.MaxPool2d(kernel_size=2))

        self.dropout = nn.Dropout(1-keep_prob)

    def forward(self, inputs):
        outputs = self.conv_block1(inputs)
        outputs = self.conv_block2(outputs)
        if self.keep_prob < 1.0:
            return self.dropout(outputs)
        else:
            return outputs

class up2D(nn.Module):
    """Up Convolution `tensor` by scale_factor times
    Args:
        tensor (4-D Tensor): (N, C H, W)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, C, 2 * H, 2 * W)
    """
    def __init__(
            self, 
           in_channels,
            n_channels, 
            is_deconv=False, 
            keep_prob=1.0,
            scale_factor=2):
        super(up2D, self).__init__()
        self.keep_prob = keep_prob
        self.scale_factor = scale_factor 
        self.is_deconv    = is_deconv 

        if is_deconv:
            self.up = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, 
                    n_channels,
                    kernel_size=scale_factor, 
                    stride=scale_factor),
                    nn.ReLU())
        else:
            self.up = nn.Sequential(
                    nn.Conv2d(in_channels, 
                        n_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1, bias=USE_BIAS),
                    nn.ReLU())
        self.dropout = nn.Dropout(1-keep_prob)
        self.upsample_model = nn.Upsample(scale_factor=self.scale_factor,mode='bilinear')

    def forward(self, inputs):
        # Apply blinear upsampling if not using deconvolution layer
        if not self.is_deconv:
            inputs = self.upsample_model(inputs)#F.interpolate(inputs, scale_factor=self.scale_factor, mode="bilinear")
        outputs = self.up(inputs)
        if self.keep_prob < 1.0:
            return self.dropout(outputs) 
        else:
            return outputs

class separable_conv2D(nn.Module):
    def __init__(self, nin=4, 
                       nout=None,
                       kernel_size=[3,3],
                       depth_multiplier=1, 
                       activation_fn=None,
                       dilation_rate=[1,1],
                       ):
        super(separable_conv2D, self).__init__()

        self.nout = nout
        self.activation_fn= activation_fn
        if isinstance(dilation_rate, tuple) or isinstance(dilation_rate, list):
            dilation_rate = list(dilation_rate)
        else:
            dilation_rate = [dilation_rate, dilation_rate]

        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            kernel_size = list(kernel_size)
        else:
            kernel_size = [kernel_size, kernel_size]

        padding = [(k-1)//2*rate for (k, rate) in zip(kernel_size, dilation_rate)]

        self.depthwise = nn.Conv2d(nin, nin * depth_multiplier, kernel_size=kernel_size, padding=padding, dilation=dilation_rate, groups=nin, bias=USE_BIAS)
        if self.nout:
            self.pointwise = nn.Conv2d(nin * depth_multiplier, self.nout, kernel_size=1, bias=USE_BIAS)

        if activation_fn:
            self.activation_fn = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        if self.nout:
            out = self.pointwise(out)
        if self.activation_fn:
            out = self.activation_fn(out)
        return out





class DWFab_block(nn.Module):
    def __init__(
            self, 
            in_channels,
            n_channels,
            kernel_size=3,
            dilation_rate=2,
            use_pool=False,
            keep_prob=1.0,
            is_batchnorm=True,
            batchnorm_type="normal",
            ):
        super(DWFab_block, self).__init__()
        self.keep_prob = keep_prob
        # First conv block - asymmetric convolution
        batchnorm_fn = choose_bn[batchnorm_type]       
        if is_batchnorm:
            self.conv_block1 = nn.Sequential(
                    separable_conv2D(in_channels, None, kernel_size=(kernel_size, 1), depth_multiplier=1),
                    nn.ReLU(),
                    separable_conv2D(in_channels, None, kernel_size=(1, kernel_size), depth_multiplier=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, n_channels, kernel_size=(1, 1), bias=USE_BIAS),
                    batchnorm_fn(n_channels),
                    nn.ReLU())
        else:
            self.conv_block1 = nn.Sequential(
                    separable_conv2D(in_channels, None, kernel_size=(kernel_size, 1), depth_multiplier=1),
                    nn.ReLU(),
                    separable_conv2D(in_channels, None, kernel_size=(1, kernel_size), depth_multiplier=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, n_channels, kernel_size=(1, 1), bias=USE_BIAS),
                    nn.ReLU())

        # Second conv block - asymmetric + dilation convolution
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2*dilation_rate,0),
																				bias=USE_BIAS,
																				dilation=(dilation_rate,1)),
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2*dilation_rate),
																				bias=USE_BIAS,
                                                                                dilation=(1, dilation_rate)),
                nn.ReLU(),
                )
        # 1x1 conv to output n_channels 
        shaping_conv    = nn.Conv2d(in_channels + n_channels, n_channels, kernel_size=1, stride=1, bias=USE_BIAS)
    
        if is_batchnorm:
            self.shaping_conv_block = nn.Sequential(
                                        shaping_conv, 
                    			batchnorm_fn(n_channels),
                                        nn.ReLU()) 
        else:
            self.shaping_conv_block = nn.Sequential(
                                        shaping_conv, 
                                        nn.ReLU()) 
            

        #Regularizer
        self.dropout = nn.Dropout(1-keep_prob)
        #Add the main branch
        self.use_pool = use_pool 
        if use_pool:
            self.maxpool2d = nn.MaxPool2d(kernel_size=2) 


    def forward(self, inputs):
        skip = inputs 
        outputs = self.conv_block1(inputs)
        outputs = self.conv_block2(outputs) 
        outputs = torch.cat([skip, outputs], 1)  
        outputs = self.shaping_conv_block(outputs)
        if self.use_pool:
            outputs = self.maxpool2d(outputs)
        if self.keep_prob < 1.0:
            return self.dropout(outputs)
        else:
            return outputs

class non_bottleneck1D(nn.Module):
    def __init__(
            self, 
            in_channels,
            n_channels,
            kernel_size=3,
            dilation_rate=2,
            use_pool=False,
            keep_prob=1.0,
            is_batchnorm=True,
            batchnorm_type="normal",
            ):
        super(non_bottleneck1D, self).__init__()
        self.keep_prob  = keep_prob
        self.in_channels = in_channels
        self.n_channels = n_channels 
        # Skip connection 
        self.skip = nn.Conv2d(in_channels, n_channels, kernel_size=(1, 1), bias=USE_BIAS)
        #First conv block - asymmetric convolution
        batchnorm_fn = choose_bn[batchnorm_type]       
        if is_batchnorm:
            self.conv_block1 = nn.Sequential(
                    nn.Conv2d(in_channels, n_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2,0), bias=USE_BIAS),
                    nn.ReLU(),
                    nn.Conv2d(n_channels, n_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=USE_BIAS),
                    batchnorm_fn(n_channels),
                    nn.ReLU())

            #Second conv block - asymmetric + dilation convolution
            self.conv_block2 = nn.Sequential(
                    nn.Conv2d(n_channels, n_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2*dilation_rate,0),
                                                                                    bias=USE_BIAS,
                                                                                    dilation=(dilation_rate,1)),
                    nn.ReLU(),
                    nn.Conv2d(n_channels, n_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2*dilation_rate),
                                                                                    bias=USE_BIAS,
                                                                                    dilation=(1, dilation_rate)),
                    batchnorm_fn(n_channels))
        else:
            self.conv_block1 = nn.Sequential(
                    nn.Conv2d(in_channels, n_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2,0), bias=USE_BIAS),
                    nn.ReLU(),
                    nn.Conv2d(n_channels, n_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=USE_BIAS),
                    nn.ReLU())

            #Second conv block - asymmetric + dilation convolution
            self.conv_block2 = nn.Sequential(
                    nn.Conv2d(n_channels, n_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2*dilation_rate,0),
                                                                                    bias=USE_BIAS,
                                                                                    dilation=(dilation_rate,1)),
                    nn.ReLU(),
                    nn.Conv2d(n_channels, n_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2*dilation_rate),
                                                                                    bias=USE_BIAS,
                                                                                    dilation=(1, dilation_rate)),
                    )
        #Regularizer
        self.dropout = nn.Dropout(1-keep_prob)
        #Add the main branch
        self.use_pool = use_pool 
        if use_pool:
            self.maxpool2d = nn.MaxPool2d(kernel_size=2) 
    def forward(self, inputs):
        if self.in_channels != self.n_channels: 
            skip    = self.skip(inputs)
        else:
            skip    = inputs 
        outputs = self.conv_block1(inputs)
        outputs = self.conv_block2(outputs) 
        outputs = skip + outputs  
        if self.use_pool:
            outputs = self.maxpool2d(outputs)
        if self.keep_prob < 1.0:
            return self.dropout(F.relu(outputs ))
        else:
            return F.relu(outputs)

