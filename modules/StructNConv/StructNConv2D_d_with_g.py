
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


class StructNConv2D_d_with_g(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1, bias=True, channel_first=False):

        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels, False, 0, groups, bias,
                                      dilation=dilation, kernel_size = (1, kernel_size, kernel_size))
        self.spatial_weight = self.weight
        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels, False, 0, groups, bias, kernel_size = (1, 1))
        self.channel_weight = self.weight

        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channel_first = channel_first

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        self.kernel_channels = KernelChannels(kernel_size, stride, padding, dilation)

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'channel_weight', pos_fn)
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_grad', pos_fn)


    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):
        gx_roll = self.kernel_channels.kernel_channels(gx)
        cgx_roll = self.kernel_channels.kernel_channel(cgx)
        gy_roll = self.kernel_channels.kernel_channels(gy)
        cgy_roll = self.kernel_channels.kernel_channels(cgy)

        gx = gx.unsqueeze(2)
        cgx = cgx.unsqueeze(2)
        gy = gy.unsqueeze(2)
        cgy = cgy.unsqueeze(2)

        gx_prop = (cgx_roll * gx_roll + cgx * gx) / (cgx_roll + cgx + self.eps)
        cgx_prop = (cgx_roll + cgx) / 2
        gy_prop = (cgy_roll * gy_roll + cgy * gy) / (cgy_roll + cgy + self.eps)
        cgy_prop = (cgy_roll + cgy) / 2

        distsx = torch.expand(
            torch.arange((self.kernel_size-1)/2, -self.kernel_size/2, -1).unsqueeze(0),
            self.kernel_size, -1).view(1, 1, -1, 1, 1) * self.dilation
        adistx = torch.abs(distsx)
        distsy = torch.expand(
            torch.arange((self.kernel_size-1)/2, -self.kernel_size/2, -1).unsqueeze(1),
            -1, self.kernel_size).view(1, 1, -1, 1, 1) * self.dilation
        adisty = torch.abs(distsy)

        g_prop = distsx * gx_prop + distsy * gy_prop
        cg_prop = (adistx * cgx_prop + adisty * cgy_prop) / (adistx + adisty+self.eps)

        d_roll = self.kernel_channels.kernel_channels(d)
        cd_roll = self.kernel_channels.kernel_channels(cd)

        d_prop = d_roll * (1+ g_prop)
        cd_prop = cd_roll * (1 + self.w_grad * cg_prop) / (1+self.w_grad)

        if self.channel_first:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cd_prop * d_prop, self.channel_weight, self.groups)
            denom = F.conv2d(cd_prop, self.channel_weight, self.groups)
            d_channel = (nom / (denom+self.eps))
            cd_channel = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along spatial dimensions
            nom = F.conv2d(cd_channel * d_channel, self.channel_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv2d(cd_channel, self.channel_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            d = nom / (denom+self.eps) + self.bias
            cd = denom / torch.sum(self.channel_weight)
        else:
            # Normalized Convolution along spatial dimensions
            nom = F.conv2d(cd_prop * d_prop, self.statial_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv2d(cd_prop, self.statial_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            d_spatial = (nom / (denom+self.eps))
            cd_spatial = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cd_spatial * d_spatial, self.channel_weight, self.groups)
            denom = F.conv2d(cd_spatial, self.channel_weight, self.groups)
            d = nom / (denom+self.eps) + self.bias
            cd = denom / torch.sum(self.channel_weight)

        return d, cd

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_grad)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)
            torch.nn.init.kaiming_uniform_(self.w_grad)
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