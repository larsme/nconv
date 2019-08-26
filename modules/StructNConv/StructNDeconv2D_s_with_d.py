
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
from modules.StructNConv.retrieve_indices import retrieve_indices

class StructNDeconv2D_s_with_d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, use_bias=True, const_bias_init=False):
        super(StructNDeconv2D_s_with_d, self).__init__()

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
        self.w_s_from_d = torch.nn.Parameter(data=torch.Tensor(2, 1, 1, 1))
        self.w_prop = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))
        self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1,
                                                                   self.kernel_size, self.kernel_size))
        if use_bias:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, self.out_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if use_bias and not const_bias_init:
                torch.nn.init.xavier_uniform_(self.bias)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_from_d)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            if use_bias and not const_bias_init:
                torch.nn.init.kaiming_uniform_(self.bias)
        if use_bias and const_bias_init:
            self.bias.data[...] = 0.01

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'channel_weight', pos_fn)
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_prop', pos_fn)

    def forward(self, d, cd, s, cs):

        # calculate smoothness from depths
        _, j_max = F.max_pool2d(d * cd, kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)
        _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)

        min_div_max = torch.abs(retrieve_indices(d, j_min) / (retrieve_indices(d, j_max) + self.eps))

        s_from_d = (1 - self.w_s_from_d[0, ...] - self.w_s_from_d[1, ...]) * min_div_max \
            + self.w_s_from_d[0, ...] * min_div_max**2 \
            + self.w_s_from_d[1, ...] * min_div_max**3
        cs_from_d = retrieve_indices(cd, j_max) * retrieve_indices(cd, j_min)

        s_prop = (self.w_prop * cs * s + 1 * cs_from_d * s_from_d) / (self.w_prop * cs + 1 * cs_from_d + self.eps)
        cs_prop = (self.w_prop * cs + 1 * cs_from_d) / (self.w_prop + 1)

        nom = F.conv_transpose2d(cs_prop * s_prop, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation)
        denom = F.conv_transpose2d(cs_prop, self.spatial_weight, groups=self.in_channels, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation)
        cdenom = F.conv_transpose2d(torch.ones_like(cs), self.spatial_weight, groups=self.in_channels,
                                    stride=self.stride, padding=self.padding, dilation=self.dilation)
        s = nom / (denom+self.eps)
        cs = denom / cdenom

        if self.use_bias:
            s += self.bias

        return s, cs
