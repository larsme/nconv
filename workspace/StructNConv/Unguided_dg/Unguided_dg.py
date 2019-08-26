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

from modules.StructNConv.StructNMaxPool2D_dg import StructNMaxPool2D_dg
from modules.StructNConv.StructNConv2D_dg import StructNConv2D_dg

from modules.StructNConv.StructNConv2D_g_with_d import StructNConv2D_g_with_d
from modules.StructNConv.StructNDeconv2D_g_with_d import StructNDeconv2D_g_with_d

from modules.StructNConv.StructNConv2D_d_with_g import StructNConv2D_d_with_g
from modules.StructNConv.StructNDeconv2D_d_with_g import StructNDeconv2D_d_with_g

from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample
from modules.StructNConv.NearestNeighbourUpsample_g_with_d import NearestNeighbourUpsample_g_with_d


class CNN(nn.modules.Module):

    def __init__(self, params):
        pos_fn = params['enforce_pos_weights']
        num_channels = params['num_channels']

        maxpool_dg = params['maxpool_dg']

        nn_upsample_g = params['nn_upsample_g']
        use_conv_bias_g = params['use_conv_bias_g']
        const_bias_init_g = params['const_bias_init_g']
        use_deconv_bias_g = params['use_deconv_bias_g']

        nn_upsample_d = params['nn_upsample_d']
        use_conv_bias_d = params['use_conv_bias_d']
        const_bias_init_d = params['const_bias_init_d']
        use_deconv_bias_d = params['use_deconv_bias_d']
        super().__init__()

        self.pos_fn = pos_fn

        # shared pooling
        if maxpool_dg:
            self.npool_dg = StructNMaxPool2D_dg(channels=num_channels,
                                                kernel_size=2, stride=2, padding=0,
                                                pos_fn=pos_fn, init_method=params['init_method'])
        else:
            self.npool_dg = StructNConv2D_dg(in_channels=num_channels, out_channels=num_channels,
                                             channel_first=False,
                                             pos_fn=pos_fn, init_method=params['init_method'],
                                             use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                             kernel_size=2, stride=2, padding=0, dilation=1)

        # gradient modules
        self.nconv1_g = StructNConv2D_g_with_d(in_channels=1, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_g = StructNConv2D_g_with_d(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_g = StructNConv2D_g_with_d(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        if nn_upsample_g:
            self.nup_g = NearestNeighbourUpsample_g_with_d(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_g = StructNDeconv2D_g_with_d(in_channels=num_channels, out_channels=num_channels,
                                                  pos_fn=pos_fn, init_method=params['init_method'],
                                                  use_bias=use_deconv_bias_g,
                                                  kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_g = StructNConv2D_g_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_g = StructNConv2D_g_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_g = StructNConv2D_g_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_g = StructNConv2D_g_with_d(in_channels=num_channels, out_channels=1,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                               kernel_size=1, stride=1, padding=0, dilation=1)


        # depth modules
        self.nconv1_d = StructNConv2D_d_with_g(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_d = StructNConv2D_d_with_g(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_d = StructNConv2D_d_with_g(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D_d_with_g(in_channels=num_channels, out_channels=num_channels,
                                                  pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_deconv_bias_d,
                                                  kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_d = StructNConv2D_d_with_g(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_d = StructNConv2D_d_with_g(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_d = StructNConv2D_d_with_g(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_d = StructNConv2D_d_with_g(in_channels=num_channels, out_channels=1,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                               kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, d_0, cd_0):
        gx_0 = cgx_0 = gy_0 = cgy_0 = torch.zeros_like(cd_0)

        # Stage 0
        gx_0, cgx_0, gy_0, cgy_0 = self.nconv1_g(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        d_0, cd_0 = self.nconv1_d(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        gx_0, cgx_0, gy_0, cgy_0 = self.nconv2_g(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        d_0, cd_0 = self.nconv2_d(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        gx_0, cgx_0, gy_0, cgy_0 = self.nconv3_g(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        d_0, cd_0 = self.nconv3_d(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)

        # Stage 1
        d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1 = self.npool_dg(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv2_g(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        d_1, cd_1 = self.nconv2_d(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv3_g(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        d_1, cd_1 = self.nconv3_d(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)

        # Stage 2
        d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2 = self.npool_dg(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        gx_2, cgx_2, gy_2, cgy_2 = self.nconv2_g(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)
        d_2, cd_2 = self.nconv2_d(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)

        # Stage 3
        d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3 = self.npool_dg(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)
        gx_3, cgx_3, gy_3, cgy_3 = self.nconv2_g( d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3)
        d_3, cd_3 = self.nconv2_d(d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3)

        # Stage 2
        gx_32, cgx_32, gy_32, cgy_32 = self.nup_g(d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3)
        d_32, cd_32 = self.nup_d(d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3)
        gx_2, cgx_2 = torch.cat((gx_32, gx_2), 1), torch.cat((cgx_32, cgx_2), 1)
        gy_2, cgy_2 = torch.cat((gy_32, gy_2), 1), torch.cat((cgy_32, cgy_2), 1)
        d_2, cd_2 = torch.cat((d_32, d_2), 1), torch.cat((cd_32, cd_2), 1)
        gx_2, cgx_2, gy_2, cgy_2 = self.nconv4_g(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)
        d_2, cd_2 = self.nconv4_d(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)

        # Stage 1
        gx_21, cgx_21, gy_21, cgy_21 = self.nup_g(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)
        d_21, cd_21 = self.nup_d(d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2)
        gx_1, cgx_1 = torch.cat((gx_21, gx_1), 1), torch.cat((cgx_21, cgx_1), 1)
        gy_1, cgy_1 = torch.cat((gy_21, gy_1), 1), torch.cat((cgy_21, cgy_1), 1)
        d_1, cd_1 = torch.cat((d_21, d_1), 1), torch.cat((cd_21, cd_1), 1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv5_g(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        d_1, cd_1 = self.nconv5_d(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)

        # Stage 0
        gx_10, cgx_10, gy_10, cgy_10 = self.nup_g(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        d_10, cd_10 = self.nup_d(d_1, cd_1, gx_1, cgx_1, gy_1, cgy_1)
        gx_0, cgx_0 = torch.cat((gx_10, gx_0), 1), torch.cat((cgx_10, cgx_0), 1)
        gy_0, cgy_0 = torch.cat((gy_10, gy_0), 1), torch.cat((cgy_10, cgy_0), 1)
        d_0, cd_0 = torch.cat((d_10, d_0), 1), torch.cat((cd_10, cd_0), 1)
        gx_0, cgx_0, gy_0, cgy_0 = self.nconv5_g(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        d_0, cd_0 = self.nconv5_d(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)

        # output
        d, cd = self.nconv7_d(d_0, cd_0, gx_0, cgx_0, gy_0, cgy_0)
        return d, cd
