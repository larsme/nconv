
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
import torch.nn as nn
import numpy as np
from scipy.stats import poisson
from scipy import signal

from modules.NConv2D import EnforcePos

class StructNMaxPool2D_s(nn.modules.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, pos_fn='softplus'):
        super(StructNMaxPool2D_s, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.pos_fn = pos_fn

        # Define Parameters
        self.w_s_pool = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))
        self.b_s_pool = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if use_bias:
                torch.nn.init.xavier_uniform_(self.bias)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_from_d)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            if use_bias:
                torch.nn.init.kaiming_uniform_(self.bias)

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'channel_weight', pos_fn)
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_prop', pos_fn)

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_pool)
            torch.nn.init.xavier_uniform_(self.b_s_pool)
        else: #elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_pool)
            torch.nn.init.kaiming_uniform_(self.b_s_pool)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):
        _, inds = F.max_pool2d(cs*(self.w_s_pool*s + self.b_s_pool), kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, return_indices=True)
        return s[inds], cs[inds] / self.stride / self.stride
