
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal

from modules.NConv2D import EnforcePos
from modules.StructNConv.KernelChannels import KernelChannels


class NearestNeighbourUpsample_s_with_d(nn.modules.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(NearestNeighbourUpsample_s_with_d, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, d, cd, s, cs, *args):
        output_size = ((d.size(2)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1,
                       (d.size(3)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1)
        s = F.interpolate(s, output_size, mode='nearest')
        cs = F.interpolate(cs, output_size, mode='nearest')
        return s, cs
