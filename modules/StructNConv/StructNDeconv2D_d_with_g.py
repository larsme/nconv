
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


class StructNDeconv2D_d_with_g(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0,
                 dilation=1, groups=1):

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

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        self.kernel_channels = KernelChannels(kernel_size, stride, padding, dilation)

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'spatial_weight', pos_fn)
            EnforcePos.apply(self, 'w_grad', pos_fn)


    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy):
        gx_roll = self.kernel_channels.deconv_kernel_channels(gx)
        cgx_roll = self.kernel_channels.deconv_kernel_channels(cgx)
        gy_roll = self.kernel_channels.deconv_kernel_channels(gy)
        cgy_roll = self.kernel_channels.deconv_kernel_channels(cgy)

        # scale gradients by kernel distance
        distsx = torch.expand(
            torch.arange((self.kernel_size-1)/2, -self.kernel_size/2, -1).unsqueeze(0),
            self.kernel_size, -1).view(1, 1, -1, 1, 1) * self.stride * self.dilation
        adistx = torch.abs(distsx)
        distsy = torch.expand(
            torch.arange((self.kernel_size-1)/2, -self.kernel_size/2, -1).unsqueeze(1),
            -1, self.kernel_size).view(1, 1, -1, 1, 1) * self.stride * self.dilation
        adisty = torch.abs(distsy)

        g_prop = distsx * gx_roll + distsy * gy_roll
        cg_prop = (adistx * cgx_roll + adisty * cgy_roll) / (adistx + adisty+self.eps)

        # propagate d, cd
        d_roll = self.kernel_channels.deconv_kernel_channels(d)
        cd_roll = self.kernel_channels.deconv_kernel_channels(cd)
        deconv_present = self.kernel_channels.deconv_kernel_channels(torch.ones_like(d))

        d_prop = d_roll * (1 + g_prop)
        cd_prop = cd_roll * (1 + self.w_grad * cg_prop) / (1+self.w_grad)

        # Normalized Deconvolution along spatial dimensions
        nom = F.conv3d(cd_prop * d_prop, self.statial_weight, self.groups)
        denom = F.conv3d(cd_prop, self.statial_weight, self.groups)
        cdenom = F.conv3d(deconv_present, self.statial_weight, self.groups)
        d = (nom / (denom+self.eps) + self.bias).squeeze(2)
        cd = (denom / (cdenom+self.eps)).squeeze(2)

        return d, cd

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_grad)
        else:  # elif self.init_method == 'k': # Kaiming
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
