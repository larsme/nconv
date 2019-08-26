
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
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, const_bias_init=False, channel_first=False):
        super(StructNConv2D_dg, self).__init__()

        self.conv_g = StructNConv2D_g_with_d(in_channels, out_channels, kernel_size,
                                             pos_fn, init_method, stride, padding,
                                             dilation, groups, use_bias, const_bias_init, channel_first)
        self.conv_d = StructNConv2D_d_with_g(in_channels, out_channels, kernel_size,
                                             pos_fn, init_method, stride, padding,
                                             dilation, groups, use_bias, const_bias_init, channel_first)

    def forward(self, d, cd, gx, cgx, gy, cgy):
        gx, cgx, gy, cgy = self.conv_d(d, cd, gx, cgx, gy, cgy)
        d, cd = self.conv_d(d, cd, gx, cgx, gy, cgy)
        return d, cd, gx, cgx, gy, cgy
