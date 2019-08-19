
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
from modules.StructNConv.StructNConv2D_gx_with_ds import StructNConv2D_gx_with_ds
from modules.StructNConv.StructNConv2D_gy_with_ds import StructNConv2D_gy_with_ds


class StructNConv2D_g_with_d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, channel_first=False):
        super(StructNConv2D_g_with_d, self).__init__()

        self.conv_gx = StructNConv2D_gx_with_ds(in_channels, out_channels, kernel_size,
                                                pos_fn, init_method, stride, padding,
                                                dilation, groups, use_bias, channel_first)
        self.conv_gy = StructNConv2D_gy_with_ds(in_channels, out_channels, kernel_size,
                                                pos_fn, init_method, stride, padding,
                                                dilation, groups, use_bias, channel_first)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):
        gx, cgx = self.conv_gx(d, cd, s, cs, gx, cgx)
        gy, cgy = self.conv_gy(d, cd, s, cs, gy, cgy)
        return gx, cgx, gy, cgy
