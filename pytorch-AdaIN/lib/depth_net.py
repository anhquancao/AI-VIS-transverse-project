import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class UpProjBlockv2(nn.Module):
    """
    Deeper Depth Prediction with Fully Convolutional Residual Networks
    """
    # branch 1: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    # branch 2: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(UpProjBlockv2, self).__init__()
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)), 
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
        
    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.pool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        block1 = self.layer1(x)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        
        return block1, block2, block3, block4
    
class R(nn.Module):
    def __init__(self, output_channel=1, output_size=(100, 100)):
        super(R, self).__init__()
        num_features = 64 + 2048 // 32
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        self.conv3 = nn.Conv2d(num_features, output_channel, kernel_size=5, padding=2, bias=True)
        self.bilinear = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        
        
        
    def forward(self, x):
        x = self.bilinear(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        
        return x
    
class MFF(nn.Module):
    def __init__(self):
        super(MFF, self).__init__()
        self.up1 = UpProjBlockv2(in_channels=256, out_channels=16)
        self.up2 = UpProjBlockv2(in_channels=512, out_channels=16)
        self.up3 = UpProjBlockv2(in_channels=1024, out_channels=16)
        self.up4 = UpProjBlockv2(in_channels=2048, out_channels=16)
        
        self.conv = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        
    def forward(self, block1, block2, block3, block4, size):
        
        m1 = self.up1(block1, size)
        
        m2 = self.up2(block2, size)
        
        m3 = self.up3(block3, size)
        
        m4 = self.up4(block4, size)
        
        x = torch.cat([m1, m2, m3, m4], 1)
        
        
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_features=2048):
        super(Decoder, self).__init__()
        self.up1 = UpProjBlockv2(num_features // 2, num_features // 4)
        self.up2 = UpProjBlockv2(num_features // 4, num_features // 8)
        self.up3 = UpProjBlockv2(num_features // 8, num_features // 16)
        self.up4 = UpProjBlockv2(num_features // 16, num_features // 32)
        self.bn = nn.BatchNorm2d(num_features // 2)
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, bias=False)
        
    def forward(self, block1, block2, block3, block4):
        x =  F.relu(self.bn(self.conv(block4)))
        x = self.up1(x, [block3.shape[2], block3.shape[3]])
        
        x = self.up2(x, [block2.shape[2], block2.shape[3]])
        
        x = self.up3(x, [block1.shape[2], block1.shape[3]])
        
        x = self.up4(x, [block1.shape[2] * 2, block1.shape[3] * 2])
        
        return x
    
class DepthV3(nn.Module):
    def __init__(self, pretrained=True, output_channel=1, output_size=(100, 100)):
        super(DepthV3, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder = Encoder()
        
        self.decoder = Decoder()
        
        self.MFF = MFF()
        self.R = R(output_channel=output_channel, output_size=output_size)
        
        self.up = UpProjBlockv2(32, 32)
        
        
    def forward(self, x):
        block1, block2, block3, block4 = self.encoder(x)
        x_decoded = self.decoder(block1, block2, block3, block4)
        x_mff = self.MFF(block1, block2, block3, block4, [x_decoded.shape[2], x_decoded.shape[3]])
        y = self.R(torch.cat([x_decoded, x_mff], 1))
        
        return y