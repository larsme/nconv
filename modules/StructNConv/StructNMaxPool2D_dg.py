
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

class StructNMaxPool2D_d(nn.module):
    def __init__(self, kernel_size, init_method='k', stride=1, padding=0, dilation=1):
        self.init_method = init_method
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        torch.nn.init.xavier_uniform_(self.wg)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):
        _, inds = F.max_pool2d(cd * (cgx + cgy + self.wg), kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)
        return d[inds], cd[inds] / self.stride / self.stride, \
               gx[inds]*self.stride, cgx[inds] / self.stride / self.stride, \
               gy[inds]*self.stride, cgy[inds] / self.stride / self.stride
