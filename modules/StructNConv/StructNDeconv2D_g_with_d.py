
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
from modules.StructNConv.StructNDeconv2D_gx_with_d import StructNDeconv2D_gx_with_d
from modules.StructNConv.StructNDeconv2D_gy_with_d import StructNDeconv2D_gy_with_d


class StructNDeconv2D_g_with_d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, const_bias_init=False):
        super(StructNDeconv2D_g_with_d, self).__init__()

        self.conv_gx = StructNDeconv2D_gx_with_d(in_channels, out_channels, kernel_size,
                                                 pos_fn, init_method, stride, padding,
                                                 dilation, groups, use_bias, const_bias_init)
        self.conv_gy = StructNDeconv2D_gy_with_d(in_channels, out_channels, kernel_size,
                                                 pos_fn, init_method, stride, padding,
                                                 dilation, groups, use_bias, const_bias_init)

    def forward(self, d, cd, gx, cgx, gy, cgy):
        gx, cgx = self.conv_gx(d, cd, gx, cgx)
        gy, cgy = self.conv_gy(d, cd, gy, cgy)
        return gx, cgx, gy, cgy
