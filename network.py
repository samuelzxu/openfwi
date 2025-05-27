# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict

import timm
from timm.models.convnext import ConvNeXtBlock
from types import MethodType

from monai.networks.blocks import UpSample, SubpixelUpsample

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNetWithBN(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNetWithBN, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x.squeeze()

# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNetSkip(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNetSkip, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4 + dim4, dim4)  # Adjusted input dim for skip connection
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3 + dim3, dim3)  # Adjusted input dim for skip connection
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2 + dim2, dim2)  # Adjusted input dim for skip connection
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1 + dim1, dim1)  # Adjusted input dim for skip connection
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x1 = self.convblock1(x) # (None, 32, 500, 70)
        x2 = self.convblock2_1(x1) # (None, 64, 250, 70)
        x2 = self.convblock2_2(x2) # (None, 64, 250, 70)
        x3 = self.convblock3_1(x2) # (None, 64, 125, 70)
        x3 = self.convblock3_2(x3) # (None, 64, 125, 70)
        x4 = self.convblock4_1(x3) # (None, 128, 63, 70) 
        x4 = self.convblock4_2(x4) # (None, 128, 63, 70)
        x5 = self.convblock5_1(x4) # (None, 128, 32, 35) 
        x5 = self.convblock5_2(x5) # (None, 128, 32, 35)
        x6 = self.convblock6_1(x5) # (None, 256, 16, 18) 
        x6 = self.convblock6_2(x6) # (None, 256, 16, 18)
        x7 = self.convblock7_1(x6) # (None, 256, 8, 9) 
        x7 = self.convblock7_2(x7) # (None, 256, 8, 9)
        x8 = self.convblock8(x7) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x8) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        
        # Skip connection with x7
        # Resize x7 if needed to match x's dimensions
        if x.size() != x7.size():
            x7_resized = F.interpolate(x7, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            x7_resized = x7
        x = torch.cat([x, x7_resized], dim=1)  # Concatenate along channel dimension
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        
        # Skip connection with x5
        if x.size() != x5.size():
            x5_resized = F.interpolate(x5, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            x5_resized = x5
        x = torch.cat([x, x5_resized], dim=1)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        
        # Skip connection with x3
        if x.size() != x3.size():
            x3_resized = F.interpolate(x3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            x3_resized = x3
        x = torch.cat([x, x3_resized], dim=1)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        
        # Skip connection with x1
        if x.size() != x1.size():
            x1_resized = F.interpolate(x1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            x1_resized = x1
        x = torch.cat([x, x1_resized], dim=1)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None, **kwargs):
        super(Conv_HPGNN, self).__init__()
        layers = [
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size, **kwargs):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 3D versions of the building blocks
NORM_LAYERS_3D = { 'bn': nn.BatchNorm3d, 'in': nn.InstanceNorm3d }

class ConvBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock3D, self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS_3D:
            layers.append(NORM_LAYERS_3D[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout3d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh3D, self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS_3D:
            layers.append(NORM_LAYERS_3D[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock3D, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS_3D:
            layers.append(NORM_LAYERS_3D[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock3D, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS_3D:
            layers.append(NORM_LAYERS_3D[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 3D version of InversionNet
class InversionNet3D(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet3D, self).__init__()
        # In the 3D model, we expect input of shape [batch, 1, 5, 1000, 70]
        # The 5 input channels in the 2D model become part of the 3D convolution dimensions
        
        # Encoder Part - adjust kernel sizes and strides for 3D
        self.convblock1 = ConvBlock3D(1, dim1, kernel_size=(3, 7, 1), stride=(1, 2, 1), padding=(1, 3, 0))
        self.convblock2_1 = ConvBlock3D(dim1, dim2, kernel_size=(3, 3, 1), stride=(1, 2, 1), padding=(1, 1, 0))
        self.convblock2_2 = ConvBlock3D(dim2, dim2, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.convblock3_1 = ConvBlock3D(dim2, dim2, kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))
        self.convblock3_2 = ConvBlock3D(dim2, dim2, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.convblock4_1 = ConvBlock3D(dim2, dim3, kernel_size=(1, 3, 1), stride=(2, 2, 1), padding=(0, 1, 0))
        self.convblock4_2 = ConvBlock3D(dim3, dim3, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.convblock5_1 = ConvBlock3D(dim3, dim3, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.convblock5_2 = ConvBlock3D(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock3D(dim3, dim4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.convblock6_2 = ConvBlock3D(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock3D(dim4, dim4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.convblock7_2 = ConvBlock3D(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = ConvBlock3D(dim4, dim5, kernel_size=(3, 8, ceil(70 * sample_spatial / 8)), padding=0)
        
        # Decoder Part
        self.deconv1_1 = DeconvBlock3D(dim5, dim5, kernel_size=(3, 5, 5))
        self.deconv1_2 = ConvBlock3D(dim5, dim5, kernel_size=3, padding=1)
        self.deconv2_1 = DeconvBlock3D(dim5, dim4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.deconv2_2 = ConvBlock3D(dim4, dim4, kernel_size=3, padding=1)
        self.deconv3_1 = DeconvBlock3D(dim4, dim3, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.deconv3_2 = ConvBlock3D(dim3, dim3, kernel_size=3, padding=1)
        self.deconv4_1 = DeconvBlock3D(dim3, dim2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.deconv4_2 = ConvBlock3D(dim2, dim2, kernel_size=3, padding=1)
        self.deconv5_1 = DeconvBlock3D(dim2, dim1, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.deconv5_2 = ConvBlock3D(dim1, dim1, kernel_size=3, padding=(0,1,1))
        
        # Final layer with padding to match exact dimensions
        self.deconv6 = ConvBlock_Tanh3D(dim1, 1, kernel_size=3, padding=(1,1,1))
        
    def forward(self, x):
        # x shape: [batch, 1, 5, 1000, 70]
        x = x.squeeze().unsqueeze(1)
        
        # Encoder Part
        x = self.convblock1(x)     # [batch, 32, 5, 500, 70]
        x = self.convblock2_1(x)   # [batch, 64, 3, 250, 70]
        x = self.convblock2_2(x)   # [batch, 64, 3, 250, 70]
        x = self.convblock3_1(x)   # [batch, 64, 3, 125, 70]
        x = self.convblock3_2(x)   # [batch, 64, 3, 125, 70]
        x = self.convblock4_1(x)   # [batch, 128, 3, 63, 70]
        x = self.convblock4_2(x)   # [batch, 128, 3, 63, 70]
        x = self.convblock5_1(x)   # [batch, 128, 3, 32, 35]
        x = self.convblock5_2(x)   # [batch, 128, 3, 32, 35]
        x = self.convblock6_1(x)   # [batch, 256, 3, 16, 18]
        x = self.convblock6_2(x)   # [batch, 256, 3, 16, 18]
        x = self.convblock7_1(x)   # [batch, 256, 3, 8, 9]
        x = self.convblock7_2(x)   # [batch, 256, 3, 8, 9]
        x = self.convblock8(x)     # [batch, 512, 1, 1, 1]
        
        # Decoder Part
        x = self.deconv1_1(x)      # [batch, 512, 1, 5, 5]
        x = self.deconv1_2(x)      # [batch, 512, 1, 5, 5]
        x = self.deconv2_1(x)      # [batch, 256, 1, 10, 10]
        x = self.deconv2_2(x)      # [batch, 256, 1, 10, 10]
        x = self.deconv3_1(x)      # [batch, 128, 1, 20, 20]
        x = self.deconv3_2(x)      # [batch, 128, 1, 20, 20]
        x = self.deconv4_1(x)      # [batch, 64, 1, 40, 40]
        x = self.deconv4_2(x)      # [batch, 64, 1, 40, 40]
        x = self.deconv5_1(x)      # [batch, 32, 1, 80, 80]
        x = self.deconv5_2(x)      # [batch, 32, 1, 80, 80]
        
        # Apply padding to match the target output size (70x70)
        x = F.pad(x, [-5, -5, -5, -5, 0, 0], mode="constant", value=0)  # [batch, 32, 1, 70, 70]
        x = self.deconv6(x)        # [batch, 1, 1, 70, 70]
        
        return x.squeeze()

# U-Net Model Definition (Formatted for Readability with Residual Blocks)

class ResidualDoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 + Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle potential channel mismatch
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # Projection shortcut: 1x1 conv + BN to match output channels
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # Store the input for the residual connection

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (without final ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut to the identity path
        identity_mapped = self.shortcut(identity)

        # Add the residual connection
        out += identity_mapped

        # Apply final ReLU
        out = self.relu(out)
        return out


class Up(nn.Module):
    """Upscaling then ResidualDoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # Input to ResidualDoubleConv = channels from upsampled layer below + channels from skip connection
            # Output of ResidualDoubleConv = desired output channels for this decoder stage
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels) # Use ResidualDoubleConv

        else: # Using ConvTranspose2d
            # ConvTranspose halves the channels: in_channels -> in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Input channels to ResidualDoubleConv
            conv_in_channels = in_channels // 2 # Channels after ConvTranspose
            skip_channels = out_channels       # Channels from skip connection
            total_in_channels = conv_in_channels + skip_channels
            self.conv = ResidualDoubleConv(total_in_channels, out_channels) # Use ResidualDoubleConv

    def forward(self, x1, x2):
        # x1 is the feature map from the layer below (needs upsampling)
        # x2 is the skip connection from the corresponding encoder layer
        x1 = self.up(x1)

        # Pad x1 if its dimensions don't match x2 after upsampling
        # Input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # Pad format: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture implementation with Residual Blocks"""

    def __init__(
        self,
        n_channels=5,
        n_classes=1,
        init_features=32,
        depth=5, # number of pooling layers
        bilinear=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList() # Store conv blocks
        self.encoder_pools = nn.ModuleList() # Store pool layers

        # Initial conv block (no pooling before it)
        # Use ResidualDoubleConv for the initial convolution block
        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs.append(self.inc)

        current_features = init_features
        for _ in range(depth):
            # Define convolution block for this stage
            conv = ResidualDoubleConv(current_features, current_features * 2)
            # Define pooling layer for this stage
            pool = nn.MaxPool2d(2)
            self.encoder_convs.append(conv)
            self.encoder_pools.append(pool)
            current_features *= 2

        # --- Bottleneck ---
        # Use ResidualDoubleConv for the bottleneck
        self.bottleneck = ResidualDoubleConv(current_features, current_features)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Input features start from bottleneck output features
        # Output features at each stage are halved
        for _ in range(depth):
            # Up block uses ResidualDoubleConv internally and handles channels
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2 # Halve features for next Up block input

        # --- Output Layer ---
        # Input features are the output features of the last Up block
        self.outc = OutConv(current_features, n_classes)

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        """Pads or crops input tensor x to target height and width."""
        _, _, h, w = x.shape
        # Pad Height if needed
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))  # Pad height only
            h = target_h
        # Pad Width if needed
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))  # Pad width only
            w = target_w
        # Crop Height if needed
        if h > target_h:
            crop_top = (h - target_h) // 2
            # Use slicing to crop
            x = x[:, :, crop_top : crop_top + target_h, :]
            h = target_h
        # Crop Width if needed
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left : crop_left + target_w]
            w = target_w
        return x

    def forward(self, x):
        # Initial pooling and resizing
        x_pooled = self.initial_pool(x)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70)

        # --- Encoder Path ---
        skip_connections = []
        xi = x_resized

        # Apply initial conv (inc)
        xi = self.encoder_convs[0](xi)
        skip_connections.append(xi) # Store output of inc

        # Apply subsequent encoder convs and pools
        # self.depth is the number of pooling layers
        for i in range(self.depth):
            # Apply conv block for this stage
            xi = self.encoder_convs[i+1](xi)
            # Store skip connection *before* pooling
            skip_connections.append(xi)
            # Apply pooling layer for this stage
            xi = self.encoder_pools[i](xi)

        # Apply bottleneck conv
        xi = self.bottleneck(xi)

        # --- Decoder Path ---
        xu = xi # Start with bottleneck output
        # Iterate through decoder blocks and corresponding skip connections in reverse
        for i, block in enumerate(self.decoder_blocks):
            # Determine the correct skip connection index from the end
            # Example: depth=5. Skips stored: [inc, enc1, enc2, enc3, enc4] (indices 0-4)
            # Decoder 0 (Up(1024, 512)) needs skip 4 (enc4)
            # Decoder 1 (Up(512, 256)) needs skip 3 (enc3) ...
            # Decoder 4 (Up(64, 32)) needs skip 0 (inc)
            skip_index = self.depth - 1 - i
            skip = skip_connections[skip_index]
            xu = block(xu, skip) # Up block combines xu (from below) and skip

        # --- Final Output ---
        logits = self.outc(xu)
        # Apply scaling and offset specific to the problem's target range
        # output = logits * 1000.0 + 1500.0
        return logits.squeeze()  # Remove the channel dimension

####################
## EMA + Ensemble ##
####################

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits= m(x)
            
            if output is None:
                output = logits
            else:
                output += logits
                
        output /= len(self.models)
        return output
        

#############
## Decoder ##
#############

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 2,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= scale_factor,
                mode= upsample_mode,
            )

        if intermediate_conv:
            k= 3
            c= skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
                )
        else:
            self.intermediate_conv= None

        self.attention1 = Attention2d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type, 
            in_channels= out_channels,
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            # print(x.shape, skip.shape)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (2,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            # print(i, ic, sc, dc)
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    attention_type= attention_type,
                    intermediate_conv= intermediate_conv,
                    upsample_mode= upsample_mode,
                    scale_factor= scale_factors[i],
                    )
            )

    def forward(self, feats: list[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        # Decoder blocks
        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = UpSample(
            spatial_dims= 2,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x
        

#############
## Encoder ##
#############

def _convnext_block_forward(self, x):
    shortcut = x
    x = self.conv_dw(x)

    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

    if self.gamma is not None:
        x = x * self.gamma.reshape(1, -1, 1, 1)

    x = self.drop_path(x) + self.shortcut(shortcut)
    return x


class CoolNet(nn.Module):
    def __init__(
        self,
        backbone: str = 'convnext_small.fb_in22k_ft_in1k',
        pretrained: bool = False,
    ):
        super().__init__()
        
        # Encoder
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= True,
            drop_path_rate=0.0,
            )
        ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1]

        # Decoder
        self.decoder= UnetDecoder2d(
            encoder_channels= ecs,
        )

        self.seg_head= SegmentationHead2d(
            in_channels= self.decoder.decoder_channels[-1],
            out_channels= 1,
            scale_factor= 1,
        )
        
        self._update_stem(backbone)
        
        self.replace_activations(self.backbone, log=True)
        self.replace_norms(self.backbone, log=True)
        self.replace_forwards(self.backbone, log=True)

    def _update_stem(self, backbone):
        if backbone.startswith("convnext"):

            # Update stride
            self.backbone.stem_0.stride = (4, 1)
            self.backbone.stem_0.padding = (0, 2)

            # Duplicate stem layer (to downsample height)
            with torch.no_grad():
                w = self.backbone.stem_0.weight
                new_conv= nn.Conv2d(w.shape[0], w.shape[0], kernel_size=(4, 4), stride=(4, 1), padding=(0, 1))
                new_conv.weight.copy_(w.repeat(1, (128//w.shape[1])+1, 1, 1)[:, :new_conv.weight.shape[1], :, :])
                new_conv.bias.copy_(self.backbone.stem_0.bias)

            self.backbone.stem_0= nn.Sequential(
                nn.ReflectionPad2d((1,1,80,80)),
                self.backbone.stem_0,
                new_conv,
            )

        else:
            raise ValueError("Custom striding not implemented.")
        pass

    def replace_activations(self, module, log=False):
        if log:
            print(f"Replacing all activations with GELU...")
        
        # Apply activations
        for name, child in module.named_children():
            if isinstance(child, (
                nn.ReLU, nn.LeakyReLU, nn.Mish, nn.Sigmoid, 
                nn.Tanh, nn.Softmax, nn.Hardtanh, nn.ELU, 
                nn.SELU, nn.PReLU, nn.CELU, nn.GELU, nn.SiLU,
            )):
                setattr(module, name, nn.GELU())
            else:
                self.replace_activations(child)

    def replace_norms(self, mod, log=False):
        if log:
            print(f"Replacing all norms with InstanceNorm...")
            
        for name, c in mod.named_children():

            # Get feature size
            n_feats= None
            if isinstance(c, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                n_feats= c.num_features
            elif isinstance(c, (nn.GroupNorm,)):
                n_feats= c.num_channels
            elif isinstance(c, (nn.LayerNorm,)):
                n_feats= c.normalized_shape[0]

            if n_feats is not None:
                new = nn.InstanceNorm2d(
                    n_feats,
                    affine=True,
                    )
                setattr(mod, name, new)
            else:
                self.replace_norms(c)

    def replace_forwards(self, mod, log=False):
        if log:
            print(f"Replacing forward functions...")
            
        for name, c in mod.named_children():
            if isinstance(c, ConvNeXtBlock):
                c.forward = MethodType(_convnext_block_forward, c)
            else:
                self.replace_forwards(c)

        
    def proc_flip(self, x_in):
        x_in= torch.flip(x_in, dims=[-3, -1])
        x= self.backbone(x_in)
        x= x[::-1]

        # Decoder
        x= self.decoder(x)
        x_seg= self.seg_head(x[-1])
        x_seg= x_seg[..., 1:-1, 1:-1]
        x_seg= torch.flip(x_seg, dims=[-1])
        return x_seg

    def forward(self, batch):
        x= batch

        # Encoder
        x_in = x
        x= self.backbone(x)
        # print([_.shape for _ in x])
        x= x[::-1]

        # Decoder
        x= self.decoder(x)
        # print([_.shape for _ in x])
        x_seg= self.seg_head(x[-1])
        x_seg= x_seg[..., 1:-1, 1:-1]
        # x_seg= x_seg * 1500 + 3000
    
        if self.training:
            return x_seg
        else:
            p1 = self.proc_flip(x_in)
            x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
            return x_seg

model_dict = {
    'InversionNet': InversionNet,
    'Discriminator': Discriminator,
    'UPFWI': FCN4_Deep_Resize_2,
    'InversionNetSkip': InversionNetSkip,
    'InversionNet3D': InversionNet3D,
    'UNet': UNet,
    'CoolNet': CoolNet,
}

