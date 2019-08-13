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

from modules.nconv import NConv2d


class exp_unguided_d(nn.Module):

    def __init__(self, pos_fn=None, num_channels=2, maxpool_d=True, maxpool_s=True):
        super().__init__() 
        
        self.pos_fn = pos_fn

        # depth modules
        self.nconv1_d = StructNConv2D_d_with_s(1, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv2_d = StructNConv2D_d_with_s(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv3_d = StructNConv2D_d_with_s(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        
        self.nconv4_d = StructNConv2D_d_with_s(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv5_d = StructNConv2D_d_with_s(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv6_d = StructNConv2D_d_with_s(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)

        self.nconv7_d = StructNConv2D_d_with_s(num_channels, 1, (1,1), pos_fn, 'k')

        if maxpool_d:
            self.npool1_d = self.npool2_d = self.npool3_d = StructNMaxPool2D_d(kernel_size=2, stride=2, padding=0)
        else:
            self.npool1_d = self.npool2_d = self.npool3_d = StructNConv2D_d_with_s(num_channels, num_channels,
                                                                                   pos_fn, 'p', kernel_size=2,
                                                                                   stride=2, padding=0)


        # border modules
        self.nconv1_s = StructNConv2D_s_with_d(1, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv2_s = StructNConv2D_s_with_d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv3_s = StructNConv2D_s_with_d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)

        self.nconv4_s = StructNConv2D_s_with_d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv5_s = StructNConv2D_s_with_s(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv6_s = StructNConv2D_s_with_d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)

        self.nconv7_s = StructNConv2D_s_with_d(num_channels, 1, (1,1), pos_fn, 'k')

        if maxpool_s:
            self.npool1_s = StructNMaxPool2D_s(kernel_size=2, stride=2, padding=0)
            self.npool2_s = StructNMaxPool2D_s(kernel_size=2, stride=2, padding=0)
            self.npool3_s = StructNMaxPool2D_s(kernel_size=2, stride=2, padding=0)
        else:
            self.npool1_s = self.npool2_s = self.npool3_s = StructNConv2D_s_with_d(num_channels, num_channels,
                                                                                   pos_fn, 'p', kernel_size=2,
                                                                                   stride=2, padding=0)
        
        
    def forward(self, d_0_0, cd_0_0):
        s_0_0 = torch.zeros_like(d_0_0)
        cs_0_0 = torch.zeros_like(cd_0_0)

        # Propagate
        s_1_0, cs_1_0 = self.nconv1_s(d_0_0, cd_0_0, s_0_0, cd_0_0)
        d_1_0, cd_1_0 = self.nconv1_d(d_0_0, cd_0_0, s_1_0)
        s_2_0, cs_2_0 = self.nconv2_s(d_1_0, cd_1_0, s_1_0, cs_1_0)
        d_2_0, cd_2_0 = self.nconv2_d(d_1_0, cd_1_0, s_2_0)
        s_3_0, cs_3_0 = self.nconv3_s(d_2_0, cd_2_0, s_2_0, cs_2_0)
        d_3_0, cd_3_0 = self.nconv3_d(d_2_0, cd_2_0, s_3_0)

        # Downsample 1
        s_0_1, cs_0_1 = self.npool1_s(d_3_0, cd_3_0, s_3_0, cs_3_0)
        d_0_1, cd_0_1 = self.npool1_s(d_3_0, cd_3_0)

        # Propagate
        s_1_1, cs_1_1 = self.nconv2_s(d_0_1, cd_0_1, s_0_1, cd_0_1)
        d_1_1, cd_1_1 = self.nconv2_d(d_0_1, cd_0_1, s_1_1)
        s_2_1, cs_2_1 = self.nconv3_s(d_1_1, cd_1_1, s_1_1, cs_1_1)
        d_2_1, cd_2_1 = self.nconv3_d(d_1_1, cd_1_1, s_2_1)
        
        # Downsample 1
        s_0_2, cs_0_2 = self.npool2_s(d_2_1, cd_2_1, s_2_1, cs_2_1)
        d_0_2, cd_0_2 = self.npool2_s(d_2_1, cd_2_1)

        # Propagate
        s_1_2, cs_1_2 = self.nconv2_s(d_0_2, cd_0_2, s_0_2, cd_0_2)
        d_1_2, cd_1_2 = self.nconv2_d(d_0_2, cd_0_2, s_1_2)

        # Downsample 3
        s_0_3, cs_0_3 = self.npool2_s(d_1_2, cd_1_2, s_1_2, cs_1_2)
        d_0_3, cd_0_3 = self.npool2_s(d_1_2, cd_1_2)

        # Propagate
        s_1_3, cs_1_3 = self.nconv2_s(d_0_3, cd_0_3, s_0_3, cd_0_3)
        d_1_3, cd_1_3 = self.nconv2_d(d_0_3, cd_0_3, s_1_3)


        # Upsample 1
        d4 = F.interpolate(d4_ds, cd3_ds.size()[2:], mode='nearest')
        cd4 = F.interpolate(cd4_ds, cd3_ds.size()[2:], mode='nearest')
        d34_ds, cd34_ds = self.nconv4(torch.cat((d3_ds,d4), 1),  torch.cat((cd3_ds,cd4), 1))
        
        # Upsample 2
        d34 = F.interpolate(d34_ds, cd2_ds.size()[2:], mode='nearest')
        cd34 = F.interpolate(cd34_ds, cd2_ds.size()[2:], mode='nearest')
        d23_ds, cd23_ds = self.nconv5(torch.cat((d2_ds,d34), 1), torch.cat((cd2_ds,cd34), 1))
        
        
        # Upsample 3
        d23 = F.interpolate(d23_ds, d0.size()[2:], mode='nearest')
        cd23 = F.interpolate(cd23_ds, cd0.size()[2:], mode='nearest')
        dout, cdout = self.nconv6(torch.cat((d23,d1), 1), torch.cat((cd23,cd1), 1))
        
        
        dout, cdout = self.nconv7(dout, cdout)
                
        return dout, cdout
        
