import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules import *
from utils import weights_init


class ConvNet(nn.Module):
    def __init__(self, planes, cfg_data, num_caps, caps_size, depth, mode):
        caps_size = 16
        super(ConvNet, self).__init__()
        channels, classes = cfg_data['channels'], cfg_data['classes']
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth
        self.mode = mode

        self.layers = nn.Sequential(
            nn.Conv2d(channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*8),
            nn.ReLU(True),
        )

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        #========= ConvCaps Layers
        for d in range(1, depth):
            if self.mode == 'EM':
                self.conv_layers.append(EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, stride=1, padding=1))
                self.norm_layers.append(nn.BatchNorm2d(4*4*num_caps))
            else:
                break

        final_shape = 4

        # EM
        if self.mode == 'EM':
            self.conv_a = nn.Conv2d(8*planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(num_caps)
            self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
            self.fc = EmRouting2d(num_caps, classes, caps_size, kernel_size=final_shape, padding=0)

        
        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)

     
        # EM
        if self.mode == 'EM':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)

            a, _ = self.fc(a, pose)
            out = a.view(a.size(0), -1)

       
        return out

