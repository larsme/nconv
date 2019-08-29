
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch

from modules.StructNConv.StructNConv2D_d_with_g import StructNConv2D_d_with_g
from modules.StructNConv.StructNConv2D_g_with_d import StructNConv2D_g_with_d


class StructNConv2D_dg(torch.nn.Module):
    def __init__(self, pos_fn='softplus', init_method='k',
                 use_bias_d=True, const_bias_init_d=False, use_bias_g=True, const_bias_init_g=False,
                 in_channels=1, out_channels=1, groups=1, channel_first=False,
                 kernel_size=1, stride=1, padding=0, dilation=1):
        super(StructNConv2D_dg, self).__init__()

        self.conv_g = StructNConv2D_g_with_d(pos_fn=pos_fn, init_method=init_method,
                                             use_bias=use_bias_g, const_bias_init=const_bias_init_g,
                                             in_channels=in_channels, out_channels=out_channels,
                                             groups=groups, channel_first=channel_first,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation)
        self.conv_d = StructNConv2D_d_with_g(pos_fn=pos_fn, init_method=init_method,
                                             use_bias=use_bias_d, const_bias_init=const_bias_init_d,
                                             in_channels=in_channels, out_channels=out_channels,
                                             groups=groups, channel_first=channel_first,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation)

    def forward(self, d, cd, gx, cgx, gy, cgy):
        gx_new, cgx_new, gy_new, cgy_new = self.conv_g(d, cd, gx, cgx, gy, cgy)
        if gx_new.shape == gx.shape:
            d, cd = self.conv_d(d, cd, gx_new, cgx_new, gy_new, cgy_new)
        else:
            d, cd = self.conv_d(d, cd, gx, cgx, gy, cgy)
        return d, cd, gx_new, cgx_new, gy_new, cgy_new
