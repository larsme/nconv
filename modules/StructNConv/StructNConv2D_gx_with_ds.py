
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


class StructNConv2D_gx_with_ds(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, const_bias_init=False, channel_first=False):
        super(StructNConv2D_gx_with_ds, self).__init__()

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
        d_left = torch.roll(d, shifts=1, dims=3)
        cd_left = torch.roll(cd, shifts=1, dims=3)
        s_left = torch.roll(s, shifts=1, dims=3)
        cd_left[:, :, :, 0] = 0

        d_right = torch.roll(d, shifts=(-1), dims=3)
        cd_right = torch.roll(cd, shifts=(-1), dims=3)
        s_right = torch.roll(s, shifts=(-1), dims=3)
        cd_right[:, :, :, -1] = 0

        cgx_from_ds = s * s_left * s_right * cd_left * cd_right
        height = (cd_left * d_left + cd_right * d_right) / (cd_left + cd_right + self.eps)
        gx_from_ds = (d_right - d_left) / 2 / (height + self.eps)

        # merge calculated gradients with propagated gradients
        gx = (self.w_prop * cgx * gx + 1 * cgx_from_ds * gx_from_ds) / \
            (self.w_prop * cgx + 1 * cgx_from_ds + self.eps)
        cgx = (self.w_prop * cgx + 1 * cgx_from_ds) / (self.w_prop + 1)

        # prepare convolution
        gx_roll = self.kernel_channels.kernel_channels(gx)
        cgx_roll = self.kernel_channels.kernel_channels(cgx)
        cgx_prop = cgx_roll * s_prod_roll


        if self.channel_first:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cgx_prop * gx_roll, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cgx_prop, self.channel_weight, groups=self.groups)
            gx_channel = (nom / (denom+self.eps))
            cgx_channel = (denom / torch.sum(self.channel_weight))

            # Normalized Convolution along spatial dimensions
            nom = F.conv3d(cgx_channel * gx_channel, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv3d(cgx_channel, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            gx = nom / (denom+self.eps)
            cgx = denom / torch.sum(self.spatial_weight)
        else:
            # Normalized Convolution along spatial dimensions
            nom = F.conv3d(cgx_prop * gx_roll, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv3d(cgx_prop, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            gx_spatial = (nom / (denom+self.eps))
            cgx_spatial = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cgx_spatial * gx_spatial, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cgx_spatial, self.channel_weight, groups=self.groups)
            gx = nom / (denom+self.eps)
            cgx = denom / torch.sum(self.channel_weight)

        if self.use_bias:
            gx += self.bias

        return gx*self.stride, cgx / self.stride / self.stride
