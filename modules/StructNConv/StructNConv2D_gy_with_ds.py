
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal

from modules.NConv2D import EnforcePos
from modules.StructNConv.KernelChannels import KernelChannels


class StructNConv2D_gy_with_ds(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, const_bias_init=False, channel_first=False):
        super(StructNConv2D_gy_with_ds, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method
        self.channel_first = channel_first
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        self.pos_fn = pos_fn
        self.kernel_channels = KernelChannels(kernel_size, stride, padding, dilation)

        # Define Parameters
        self.w_prop = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))
        if self.channel_first:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels,
                                                                       1, 1, 1))
            self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1,
                                                                       self.kernel_size**2, 1, 1))
        else:
            self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1,
                                                                       self.kernel_size**2, 1, 1))
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels,
                                                                       1, 1))
        if use_bias:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, self.out_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            if use_bias and not const_bias_init:
                torch.nn.init.xavier_uniform_(self.bias)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            if use_bias and not const_bias_init:
                torch.nn.init.kaiming_uniform_(self.bias)
        if use_bias and const_bias_init:
            self.bias.data[...] = 0.01

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'channel_weight', pos_fn)
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_prop', pos_fn)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy, s_prod_roll):

        # calculate gradients from depths
        d_up = torch.roll(d, shifts=1, dims=2)
        cd_up = torch.roll(cd, shifts=1, dims=2)
        s_up = torch.roll(s, shifts=1, dims=2)
        cd_up[:, :, 0, :] = 0

        d_down = torch.roll(d, shifts=(-1), dims=2)
        cd_down = torch.roll(cd, shifts=(-1), dims=2)
        s_down = torch.roll(s, shifts=(-1), dims=2)
        cd_down[:, :, -1, :] = 0

        cgy_from_ds = s * s_up * s_down * cd_up * cd_up
        height = (cd_up * d_up + cd_down * d_down) / (cd_up + cd_down + self.eps)
        gy_from_ds = (d_down - d_up) / 2 / (height + self.eps)

        # merge calculated gradients with propagated gradients
        gy = (self.w_prop * cgy * gy + 1 * cgy_from_ds * gy_from_ds) / \
            (self.w_prop * cgy + 1 * cgy_from_ds + self.eps)
        cgy = (self.w_prop * cgy + 1 * cgy_from_ds) / (self.w_prop + 1)

        # prepare convolution
        gy_roll = self.kernel_channels.kernel_channels(gy)
        cgy_roll = self.kernel_channels.kernel_channels(cgy)
        cgy_prop = cgy_roll * s_prod_roll

        if self.channel_first:
            # Normalized Convolution along channel dimensions
            nom = F.conv3d(cgy_prop * gy_roll, self.channel_weight, groups=self.groups)
            denom = F.conv3d(cgy_prop, self.channel_weight, groups=self.groups)
            gy_channel = (nom / (denom+self.eps))
            cgy_channel = (denom / torch.sum(self.channel_weight))

            # Normalized Convolution along spatial dimensions
            nom = F.conv3d(cgy_channel * gy_channel, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv3d(cgy_channel, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            gy = nom / (denom+self.eps)
            cgy = denom / torch.sum(self.spatial_weight)
        else:
            # Normalized Convolution along spatial dimensions
            nom = F.conv3d(cgy_prop * gy_roll, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv3d(cgy_prop, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            gy_spatial = (nom / (denom+self.eps))
            cgy_spatial = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cgy_spatial * gy_spatial, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cgy_spatial, self.channel_weight, groups=self.groups)
            gy = nom / (denom+self.eps)
            cgy = denom / torch.sum(self.channel_weight)

        if self.use_bias:
            gy += self.bias

        return gy*self.stride, cgy / self.stride / self.stride
