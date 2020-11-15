
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F

from modules.NConv2D import EnforcePos
from modules.StructNConv.KernelChannels import KernelChannels


class StructNConv2D_d_with_s(torch.nn.Module):
    def __init__(self, init_method='k', in_channels=1, out_channels=1, groups=1,
                 kernel_size=1, stride=1, padding=0, dilation=1, devalue_pooled_confidence=True):
        super(StructNConv2D_d_with_s, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1

        self.kernel_channels = KernelChannels(kernel_size, stride, padding, dilation)

        self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size**2, 1, 1))
        self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)


    def enforce_limits():
        # Enforce positive weights
        self.channel_weight.data = F.softplus(self.channel_weight, beta=10)
        self.spatial_weight.data = F.softplus(self.spatial_weight, beta=10)

    def forward(self, d, cd, s, cs, s_prod_roll):

        d_roll = self.kernel_channels.kernel_channels(d)
        cd_roll = self.kernel_channels.kernel_channels(cd)
        cd_prop = cd_roll * s_prod_roll

        # Normalized Convolution along spatial dimensions
        nom = F.conv3d(cd_prop * d_roll, self.spatial_weight, groups=self.in_channels).squeeze(2)
        denom = F.conv3d(cd_prop, self.spatial_weight, groups=self.in_channels).squeeze(2)
        d_spatial = nom / (denom+self.eps)
        cd_spatial = denom / (torch.sum(self.spatial_weight)+self.eps)

        # Normalized Convolution along channel dimensions
        nom = F.conv2d(cd_spatial * d_spatial, self.channel_weight, groups=self.groups)
        denom = F.conv2d(cd_spatial, self.channel_weight, groups=self.groups)
        d = nom / (denom+self.eps)
        cd = denom / (torch.sum(self.channel_weight)+self.eps)

        return d, cd * self.devalue_conf
