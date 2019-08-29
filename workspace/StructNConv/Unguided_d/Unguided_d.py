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


class CNN(nn.modules.Module):

    def __init__(self, params):
        pos_fn = params['enforce_pos_weights']
        num_channels = params['num_channels']
        maxpool_d = params['maxpool_d']
        nn_upsample_d = params['nn_upsample_d']
        use_conv_bias_d = params['use_conv_bias_d']
        use_deconv_bias_d = params['use_deconv_bias_d']
        const_bias_init_d = params['const_bias_init_d']
        super().__init__() 
        
        self.pos_fn = pos_fn

        # depth modules
        self.nconv1_d = StructNConv2D_d(in_channels=1, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=2, dilation=1)
        if maxpool_d:
            self.npool_d = StructNMaxPool2D_d(kernel_size=2, stride=2, padding=0)
        else:
            self.npool_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                           pos_fn=pos_fn, init_method=params['init_method'],
                                           use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                           kernel_size=2, stride=2, padding=0, dilation=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D_d(in_channels=num_channels, out_channels=num_channels,
                                           pos_fn=pos_fn, init_method=params['init_method'], use_bias=use_deconv_bias_d,
                                           kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_d = StructNConv2D_d(in_channels=num_channels, out_channels=1,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, d_0, cd_0):
        assert d_0.shape[3] % (self.nup_d.kernel_size**3) == 0
        assert d_0.shape == cd_0.shape

        # Stage 0
        d_0, cd_0 = self.nconv1_d(d_0, cd_0)
        d_0, cd_0 = self.nconv2_d(d_0, cd_0)
        d_0, cd_0 = self.nconv3_d(d_0, cd_0)

        # Stage 1
        d_1, cd_1 = self.npool_d(d_0, cd_0)
        d_1, cd_1 = self.nconv2_d(d_1, cd_1)
        d_1, cd_1 = self.nconv3_d(d_1, cd_1)

        # Stage 2
        d_2, cd_2 = self.npool_d(d_1, cd_1)
        d_2, cd_2 = self.nconv2_d(d_2, cd_2)

        # Stage 3
        d_3, cd_3 = self.npool_d(d_2, cd_2)
        d_3, cd_3 = self.nconv2_d(d_3, cd_3)

        # Stage 2
        d_32, cd_32 = self.nup_d(d_3, cd_3)
        d_2, cd_2 = self.nconv4_d(torch.cat((d_32, d_2), 1),  torch.cat((cd_32, cd_2), 1))
        
        # Stage 1
        d_21, cd_21 = self.nup_d(d_2, cd_2)
        d_1, cd_1 = self.nconv5_d(torch.cat((d_21, d_1), 1),  torch.cat((cd_21, cd_1), 1))

        # Stage 0
        d_10, cd_10 = self.nup_d(d_1, cd_1)
        d_0, cd_0 = self.nconv6_d(torch.cat((d_10, d_0), 1),  torch.cat((cd_10, cd_0), 1))
        
        # output
        d, cd = self.nconv7_d(d_0, cd_0)
        return d, cd
