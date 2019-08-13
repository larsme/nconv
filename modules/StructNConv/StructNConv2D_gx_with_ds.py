
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
from modules.StructNConv.KernelRoll import KernelRoll


class StructNConv2d_gx(_ConvNd):
    def __init__(self, in_channels, out_channels, pos_fn='softplus', init_method='k', groups=1, bias=True):

        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels, False, 0, groups, bias,
                                      stride=1, padding=1, dilation=1, kernel_size = 3)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)
            EnforcePos.apply(self, 'w_prop', pos_fn)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):

        # calculate gradients from depths
        d_left = torch.roll(d, shifts=(1), dims=(3))
        cd_left = torch.roll(cd, shifts=(1), dims=(3))
        s_left = torch.roll(s, shifts=(1), dims=(3))
        cd_left[:, :, :, 0] = 0

        d_right = torch.roll(d, shifts=(-1), dims=(3))
        cd_right = torch.roll(cd, shifts=(-1), dims=(3))
        s_right = torch.roll(s, shifts=(-1), dims=(3))
        cd_right[:, :, :, -1] = 0

        cgx_from_ds = s * s_left * s_right * cd_left * cd_right
        height = (cd_left * d_left + cd_right * d_right) / (cd_left + cd_right)
        gx_from_ds = (d_right - d_left) / 2 / height

        gx = (self.w_prop * cgx * gx + 1 * cgx_from_ds * gx_from_ds) / \
            (self.w_prop * cgx + 1 * cgx_from_ds)
        cgx = (self.w_prop * cgx + 1 * cgx_from_ds) / (self.w_prop + 1)

        # Normalized Convolution with border masking during propagation
        denom = F.conv2d(cgx*s, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups) \
            + cgx * (1-s) * self.weight[self.kernel_size/2, self.kernel_size/2]
        nom = F.conv2d(gx*cgx*s, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups) \
            + gx * cgx * (1-s) * self.weight[self.kernel_size/2, self.kernel_size/2]
        nconv = nom / (denom+self.eps)+ self.bias
        cout = denom / torch.sum(self.weight)

        return nconv*self.stride, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.kaiming_uniform_(self.w_prop)
        # elif self.init_method == 'p': # Poisson
        #     mu=self.kernel_size[0]/2
        #     dist = poisson(mu)
        #     x = np.arange(0, self.kernel_size[0])
        #     y = np.expand_dims(dist.pmf(x),1)
        #     w = signal.convolve2d(y, y.transpose(), 'full')
        #     w = torch.Tensor(w).type_as(self.weight)
        #     w = torch.unsqueeze(w,0)
        #     w = torch.unsqueeze(w,1)
        #     w = w.repeat(self.out_channels, 1, 1, 1)
        #     w = w.repeat(1, self.in_channels, 1, 1)
        #     self.weight.data = w + torch.rand(w.shape)

        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels)+0.01)
