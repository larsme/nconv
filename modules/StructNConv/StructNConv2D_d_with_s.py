
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


class StructNConv2d_d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, bias=True):

        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels,
                                      kernel_size, stride, padding, dilation, False, 0, groups, bias)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        self.kernel_roll = KernelRoll(kernel_size, stride, padding, dilation)

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):

        d_roll = self.kernel_roll.kernel_channels(d)
        cd_roll = self.kernel_roll.kernel_channels(cd)
        s_prod_roll, cs_prod_roll = self.kernel_roll.s_prod_kernel_channels(s, cs)
        cd_prop = cd_roll * s_prod_roll

        # Normalized Convolution along spatial dimensions
        nom = F.conv3d(cd_prop * d_roll, self.statial_weight, self.groups)
        denom = F.conv3d(cd_prop, self.statial_weight, self.groups)
        d_spatial = (nom / (denom+self.eps) + self.bias).squeeze(2)
        cd_spatial = (denom / torch.sum(self.spatial_weight)).squeeze(2)

        # Normalized Convolution along spatial dimensions
        nom = F.conv3d(cd_spatial * d_spatial, self.channel_weight, self.groups)
        denom = F.conv3d(cd_spatial, self.channel_weight, self.groups)
        d = nom / (denom+self.eps)
        cd = denom / torch.sum(self.channel_weight)

        return d, cd

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.w_s_from_d)
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