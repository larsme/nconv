
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


class StructNConv2D_g_with_ds(torch.nn.Module):
    def __init__(self, pos_fn='softplus', init_method='k', use_bias=True, const_bias_init=False,
                 in_channels=1, out_channels=1, groups=1, channel_first=False,
                 kernel_size=1, stride=1, padding=0, dilation=1, devalue_pooled_confidence=True):
        super(StructNConv2D_g_with_ds, self).__init__()

        self.conv_gx = StructNConv2D_gx_with_ds(pos_fn=pos_fn, init_method=init_method,
                                                use_bias=use_bias, const_bias_init=const_bias_init,
                                                in_channels=in_channels, out_channels=out_channels,
                                                groups=groups, channel_first=channel_first,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation,
                                                devalue_pooled_confidence=devalue_pooled_confidence)
        self.conv_gy = StructNConv2D_gy_with_ds(pos_fn=pos_fn, init_method=init_method,
                                                use_bias=use_bias, const_bias_init=const_bias_init,
                                                in_channels=in_channels, out_channels=out_channels,
                                                groups=groups, channel_first=channel_first,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation,
                                                devalue_pooled_confidence=devalue_pooled_confidence)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy, s_prod):
        gx, cgx = self.conv_gx(d, cd, s, cs, gx, cgx, s_prod)
        gy, cgy = self.conv_gy(d, cd, s, cs, gy, cgy, s_prod)
        return gx, cgx, gy, cgy
