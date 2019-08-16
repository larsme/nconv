########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.StructNConv.StructNConv2D_d import StructNConv2D_d
from modules.StructNConv.StructNMaxPool2D_d import StructNMaxPool2D_d
from modules.StructNConv.StructNDeconv2D_d import StructNDeconv2D_d
from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample


class Unguided_d(nn.modules.Module):

    def __init__(self, pos_fn=None, num_channels=5, maxpool_d=False, nn_upsample_d=False):
        super().__init__() 
        
        self.pos_fn = pos_fn

        # depth modules
        self.nconv1_d = StructNConv2D_d(1, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv2_d = StructNConv2D_d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv3_d = StructNConv2D_d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)

        if maxpool_d:
            self.npool_d = StructNMaxPool2D_d(kernel_size=2, stride=2, padding=0)
        else:
            self.npool_d = StructNConv2D_d(num_channels, num_channels,pos_fn=pos_fn, init_method='p',
                                           kernel_size=2, stride=2, padding=0)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D_d(num_channels, num_channels, pos_fn=pos_fn, init_method='p',
                                           kernel_size=2, stride=2, padding=0)

        self.nconv4_d = StructNConv2D_d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1,
                                        channel_first=True)
        self.nconv5_d = StructNConv2D_d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1,
                                        channel_first=True)
        self.nconv6_d = StructNConv2D_d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1,
                                        channel_first=True)

        self.nconv7_d = StructNConv2D_d(num_channels, 1, (1,1), pos_fn, 'k')

    def forward(self, d_0_0, cd_0_0):

        # Stage 0
        d_1_0, cd_1_0 = self.nconv1_d(d_0_0, cd_0_0)
        d_2_0, cd_2_0 = self.nconv2_d(d_1_0, cd_1_0)
        d_3_0, cd_3_0 = self.nconv3_d(d_2_0, cd_2_0)

        # Stage 1
        d_0_1, cd_0_1 = self.npool_s(d_3_0, cd_3_0)
        d_1_1, cd_1_1 = self.nconv2_d(d_0_1, cd_0_1)
        d_2_1, cd_2_1 = self.nconv3_d(d_1_1, cd_1_1)

        # Stage 2
        d_0_2, cd_0_2 = self.npool_s(d_2_1, cd_2_1)
        d_1_2, cd_1_2 = self.nconv2_d(d_0_2, cd_0_2)

        # Stage 3
        d_0_3, cd_0_3 = self.npool_s(d_1_2, cd_1_2)
        d_1_3, cd_1_3 = self.nconv2_d(d_0_3, cd_0_3)

        # Stage 2
        d_0_2_1, cd_0_2_1 = self.nup_d(d_1_3, cd_1_3)
        d_2_2, cd_2_2 = self.nconv4_d(torch.cat((d_0_2_1, d_1_2), 1),  torch.cat((cd_0_2_1, cd_1_2), 1))
        
        # Stage 1
        d_0_1_1, cd_0_1_1 = self.nup_d(d_2_2, cd_2_2)
        d_3_1, cd_3_1 = self.nconv5_d(torch.cat((d_0_1_1, d_2_1), 1),  torch.cat((cd_0_1_1, cd_2_1), 1))

        # Stage 0
        d_0_0_1, cd_0_0_1 = self.nup_d(d_3_1, cd_3_1)
        d_4_0, cd_4_0 = self.nconv_d(torch.cat((d_0_0_1, d_3_0), 1),  torch.cat((cd_0_0_1, cd_3_0), 1))
        
        # output
        d, cd = self.nconv7_d(d_4_0, cd_4_0)
        return d, cd
