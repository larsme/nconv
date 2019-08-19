
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


class StructNDeconv2D_d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=False):
        super(StructNDeconv2D_d, self).__init__()

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
        self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1,
                                                                   self.kernel_size, self.kernel_size))
        if use_bias:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, self.out_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            if use_bias:
                torch.nn.init.xavier_uniform_(self.bias)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            if use_bias:
                torch.nn.init.kaiming_uniform_(self.bias)

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'spatial_weight', pos_fn)

    def forward(self, d, cd):
        # Normalized Deconvolution along spatial dimensions
        nom = F.conv_transpose2d(cd * d, self.spatial_weight, groups=self.in_channels,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation)
        denom = F.conv_transpose2d(cd, self.spatial_weight, groups=self.in_channels,
                                   stride=self.stride, padding=self.padding, dilation=self.dilation)
        cdenom = F.conv_transpose2d(torch.ones_like(cd), self.spatial_weight, groups=self.in_channels,
                                    stride=self.stride, padding=self.padding, dilation=self.dilation)
        d = nom / (denom+self.eps)
        cd = denom / (cdenom+self.eps)

        if self.use_bias:
            d += self.bias

        return d, cd
