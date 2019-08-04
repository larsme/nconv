
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


class StructNConv2d_d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=1, padding=0, dilation=1, groups=1, bias=True):

        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels,
                                      kernel_size, stride, padding, dilation, False, 0, groups, bias)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, d, cd, s, sc):

        # Normalized Convolution with border masking during propagation
        denom = (F.conv2d(cd*s, self.weight, None, self.stride,
                          self.padding, self.dilation, self.groups)
                 + cd * (1-s) * self.weight[1, 1]) * s
        nomin = (F.conv2d(d*cd*s, self.weight, None, self.stride,
                          self.padding, self.dilation, self.groups)
                 + d * cd * (1-s) * self.weight[1, 1]) * s
        nconv = nomin / (denom+self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / s
        cout = cout.view(sz)

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
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
