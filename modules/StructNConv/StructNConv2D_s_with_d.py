
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

class StructNConv2d_s_with_d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k',
                 stride=1, padding=0, dilation=1, groups=1, bias=True, channel_first=False):
        
        # Call _ConvNd constructor
        super(_ConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                      False, 0, groups, bias)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        self.channel_first = channel_first
        
        # Initialize weights and bias
        self.init_parameters()

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

        min_div_max = torch.abs(d[j_max] / (d[j_min] + self.eps))

        s_from_d = (1 - self.w_s_from_d[0] - self.w_s_from_d[1]) * min_div_max \
            + self.w_s_from_d[0] * min_div_max**2 \
            + self.w_s_from_d[1] * min_div_max**3
        cs_from_d = cd[j_max] * cd[j_min]

        s_prop = (self.w_prop * cs * s + 1 * cs_from_d * s_from_d) / (self.w_prop * cs + 1 * cs_from_d)
        cs_prop = (self.w_prop * cs + 1 * cs_from_d) / (self.w_prop + 1)

        if self.channel_first:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cs_prop * s_prop, self.channel_weight, self.groups)
            denom = F.conv2d(cs_prop, self.channel_weight, self.groups)
            s_channel = (nom / (denom+self.eps))
            cs_channel = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along spatial dimensions
            nom = F.conv2d(cs_channel * s_channel, self.channel_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv2d(cs_channel, self.channel_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            s = nom / (denom+self.eps) + self.bias
            cs = denom / torch.sum(self.channel_weight)
        else:
            # Normalized Convolution along spatial dimensions
            nom = F.conv2d(cs_prop * s_prop, self.statial_weight, groups=self.in_channels, stride=self.stride,
                           padding=self.padding, dilation=self.dilation).squeeze(2)
            denom = F.conv2d(cs_prop, self.statial_weight, groups=self.in_channels, stride=self.stride,
                             padding=self.padding, dilation=self.dilation).squeeze(2)
            s_spatial = (nom / (denom+self.eps))
            cs_spatial = (denom / torch.sum(self.spatial_weight))

            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cs_spatial * s_spatial, self.channel_weight, self.groups)
            denom = F.conv2d(cs_spatial, self.channel_weight, self.groups)
            s = nom / (denom+self.eps) + self.bias
            cs = denom / torch.sum(self.channel_weight)

        return s, cs

    
    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier
            torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(self.spatial_weight)
            torch.nn.init.xavier_uniform_(self.w_prop)
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
        else: #elif self.init_method == 'k': # Kaiming
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


# # Non-negativity enforcement class        
# class EnforceMonotonity(object):
#     def __init__(self):
#         #TODO


#     @staticmethod
#     def apply(module):
#         fn = EnforceMonotonity()
        
#         module.register_forward_pre_hook(fn)                    

#         return fn

#     def __call__(self, module, inputs):
#        if module.training:
#             w_s_from_d = getattr(module, "w_s_from_d")
#             w_s_from_d.data = self._monoton(w_s_from_d).data
#        else:
#             pass

#     def _monoton(self, w_s_from_d):
#         # https://www.wolframalpha.com/input/?x=0&y=0&i=solve+(((1-a-b)+%2B+2*a*x+%2B+3*b*x%5E2+%3E%3D++0),+a)