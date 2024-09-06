import torch
import numpy as np
import torch.nn as nn
from models.s4.s4d import S4D
import os
import sys
import torchsummary
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'models'))

def conv3x3(in_planes,out_planes,dims,stride=1,groups=1,dilation=1):
    """3x3 or 9x1 convolution with padding"""
    if dims == 1:
        return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=int(np.floor(9/2)), groups=groups, bias=False, dilation=dilation)
    elif dims == 2: 
        return nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), stride=(stride,1),
                     padding=(dilation,0), groups=groups, bias=False, dilation=dilation)
    else:
        raise ValueError('Dims not recognised') 

def conv1x1(in_planes, out_planes, dims, stride=1):
    """1x1 or 1x1 convolution"""
    if dims == 1:
        return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    elif dims == 2:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(stride,1), bias=False)
    else:
        raise ValueError('Dims not recognised')

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, dims, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        # Generalise to dim = 1 or 2
        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes,planes,dims=dims,stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes,dims=dims)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], dims=2, out_dim=512, in_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        self.output_dim = out_dim
        # Create exception for weird input combinations
        if dims == 1 and in_channels!=1:
            raise ValueError('1-d input only supports 1 input channel')

        if norm_layer is None:
            if dims == 1:
                norm_layer = nn.BatchNorm1d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            else:
                raise ValueError('Dims not recognised')
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False,False,False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if dims == 1:
            self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=49, stride=2, padding=3,
                                bias=False)
            self.maxpool = nn.MaxPool1d(kernel_size=9, stride=2, padding=1)         
        
        elif dims == 2:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=(7,1), stride=(2,1), padding=(3,0),
                                bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1), padding=(1,0))
        else:
            raise ValueError('Dims not recognised')
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block,64,layers[0],dims=dims)
        self.layer2 = self._make_layer(block,128,layers[1],dims=dims,stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,256,layers[2],dims=dims,stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,512,layers[3],dims=dims,stride=2,dilate=replace_stride_with_dilation[2])
        
        if dims == 1:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        elif dims == 2:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        else:
            raise ValueError('Dims not recognised')
        
        self.fc = nn.Linear(512,out_dim)

        if dims == 1:
            for m in self.modules():
                if isinstance(m,nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                elif isinstance(m,(nn.BatchNorm1d,nn.GroupNorm)):
                    nn.init.constant_(m.weight,1)
                    nn.init.constant_(m.bias,0)
        elif dims == 2:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError('Dims not recognised')
        
        #zero initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m,Bottleneck):
                #     nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self,block,planes,blocks,dims,stride=1,dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilate *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, dims, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, dims, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dims, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _forward_impl(self,x):#x with shape[128,3,128,60]
        x = self.conv1(x)#[128,64,64,60]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#[128,64,32,60]

        x = self.layer1(x)#[128,64,32,60]
        x = self.layer2(x)#[128,128,16,60]
        x = self.layer3(x)#[128,256,8,60]
        x = self.layer4(x)#[128,512,4,60]

        #x = self.avgpool(x)
        x = x.mean(dim=2)#[128,512,60]

        #x = torch.flatten(x,1)
        #x = self.fc(x)
        return x
    def forward(self,x):
        return self._forward_impl(x)

class S4Model_Resnet(nn.Module):

    def __init__(
        self,
        pt,
        d_input=1,
        lr = 0.001,
        d_output=10,
        d_model=512,
        n_layers=8,
        dropout=0.2,
        prenorm=False,
        dropout_fn=nn.Dropout,
        in_channels=3#in_channels=3
    ):
        super().__init__()

        self.prenorm = prenorm
        self.resnet = ResNet(block=BasicBlock,layers=[2,2,2,2],dims=2,out_dim=512,in_channels=in_channels)
        
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                #S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model)) 
            self.dropouts.append(dropout_fn(dropout))
        
        #cont 
        self.contlayer1 = nn.Sequential(
            nn.Linear(512,512),
            nn.PReLU(512)
        )
        self.contlayer2 = nn.Linear(512,256)
        self.bilinear_product_weight = nn.Parameter(torch.randn(256,256))
        self.pt = pt
    def forward(self,x):
        B = x.shape[0]
        x = self.resnet(x) #(B,L)#[128,3,128,60]->[128,512,60]
        #x = x.unsqueeze(1)
        i = 0
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)#(B,H,L)
        #temporary removed
        # if self.pt:
        #     x2 = x.mean(dim=2)#(B,L)(16,512)#either 16 or 128 above is the batch size, which can be changed
        #     x_cont = self.contlayer1(x2)#(16,512)->(16,512)
        #     x_cont = self.contlayer2(x_cont)#(16,256)
        #     embed_sound,embed_positive = torch.split(x_cont, B//2 ,dim=0)
        #     projection_positive = torch.matmul(self.bilinear_product_weight,embed_positive.t())
        #     cont_output = torch.matmul(embed_sound,projection_positive)#(8,8)
        #     return cont_output
        # print('hello')
        return x

class S4Model_FTResnet(nn.Module):
    def __init__(
        self,
        pt_path,
        d_input,
        lr = 0.001,
        d_output=5,
        d_model=512,
        n_layers=8,
        dropout=0.2,
        prenorm=False,
        dropout_fn=nn.Dropout,
        in_channels=3
    ):
        super().__init__()
        self.prenorm = prenorm
        pt = None
        self.ss = S4Model_Resnet(d_input, pt, lr=lr, d_output=d_output, d_model=d_model, n_layers=n_layers, dropout=dropout, prenorm=prenorm, dropout_fn=dropout_fn,in_channels=3)
        self.ss.load_state_dict(torch.load(pt_path)['model'])
        
        
        #self.decoder = nn.Linear(512,d_output)
        self.decoder0 = nn.Linear(512,128)
        self.decoder = nn.Linear(128,128)
        self.drop0 = nn.Dropout(p=0.1)
        self.drop1 = nn.Dropout(p=0.1)
        self.act_fn = torch.nn.functional.relu
        self.decoder1 = nn.Linear(128,d_output)
    def freeze_ss(self):
        for param in self.ss.parameters():
            param.requires_grad = False
    
    def unfreeze_ss(self):
        for param in self.ss.parameters():
            param.requires_grad = True
    
    def forward(self,x):
        x = self.ss(x)
     
        #x = x.squeeze(1)

        x = x.mean(dim=2)
        
        #x = self.decoder(x)
        x = self.decoder0(x)
        x = self.act_fn(self.drop0(x))
        x = self.decoder(x)
        x = self.act_fn(self.drop1(x))
        x = self.decoder1(x)

        return x

