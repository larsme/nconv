
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


class StructNDeconv2D_gx_with_d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=False):
        super(StructNDeconv2D_gx_with_d, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        self.pos_fn = pos_fn

        # Define Parameters
        self.w_prop = torch.nn.Parameter(data=torch.zeros(1, self.in_channels, 1, 1))
        self.spatial_weight = torch.nn.Parameter(data=torch.zeros(self.in_channels, 1,
                                                                   self.kernel_size, self.kernel_size))
        if use_bias:
            self.bias = torch.nn.Parameter(data=torch.zeros(1, self.out_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if use_bias:
                torch.nn.init.xavier_uniform_(self.bias)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if use_bias:
                torch.nn.init.kaiming_uniform_(self.bias)

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_prop', pos_fn)

    def forward(self, d, cd, gx, cgx):

        # calculate gradients from depths
        d_left = torch.roll(d, shifts=1, dims=3)
        cd_left = torch.roll(cd, shifts=1, dims=3)
        cd_left[:, :, :, 0] = 0

        d_right = torch.roll(d, shifts=(-1), dims=3)
        cd_right = torch.roll(cd, shifts=(-1), dims=3)
        cd_right[:, :, :, -1] = 0

        cgx_from_ds = cd_left * cd_right
        height = (cd_left * d_left + cd_right * d_right) / (cd_left + cd_right)
        gx_from_ds = (d_right - d_left) / 2 / height

        # merge calculated gradients with propagated gradients
        gx = (self.w_prop * cgx * gx + 1 * cgx_from_ds * gx_from_ds) / \
            (self.w_prop * cgx + 1 * cgx_from_ds)
        cgx = (self.w_prop * cgx + 1 * cgx_from_ds) / (self.w_prop + 1)

        # Normalized Deconvolution along spatial dimensions
        nom = F.conv_transpose2d(cgx * gx, self.spatial_weight, groups=self.in_channels,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation)
        denom = F.conv_transpose2d(cgx, self.spatial_weight, groups=self.in_channels,
                                   stride=self.stride, padding=self.padding, dilation=self.dilation)
        cdenom = F.conv_transpose2d(torch.ones_like(cgx), self.spatial_weight, groups=self.in_channels,
                                    stride=self.stride, padding=self.padding, dilation=self.dilation)
        gx = (nom / (denom+self.eps) + self.bias)
        cgx = (denom / (cdenom+self.eps))

        if self.use_bias:
            gx += self.bias

        return gx/self.stride, cgx
