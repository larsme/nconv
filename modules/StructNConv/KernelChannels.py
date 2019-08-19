
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal

from modules.NConv2D import EnforcePos


class KernelChannels(nn.modules.Module):
    def __init__(self,  kernel_size, stride=1, padding=0, dilation=1):
        super(KernelChannels, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def kernel_channels(self, x):
        '''
        :param x: Input
        input to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        '''

        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        x_res = torch.Tensor(x.size(0), x.size(1), self.kernel_size**2,
                             int((x.size(2)-(self.dilation*self.kernel_size-1)-1)/self.stride+1),
                             int((x.size(3)-(self.dilation*self.kernel_size-1)-1)/self.stride+1))

        for down_offset in range(0, self.kernel_size*self.dilation, self.dilation):
            for right_offset in range(0, self.kernel_size*self.dilation, self.dilation):
                x_res[:, :, down_offset*self.kernel_size + right_offset, :, :] \
                    = x[:, :, down_offset: x_res.size(3) * self.stride + down_offset: self.stride,
                        right_offset: x_res.size(4) * self.stride + right_offset: self.stride]

        return x_res

    def deconv_kernel_channels(self, x):
        '''
        :param x: Input
        input to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        '''

        x = x.unsqueeze(2)
        x_res = torch.zeros(x.size(0), x.size(1), 0,
                                 (x.size(3)-1)*self.stride-(self.kernel_size-1)*self.dilation+1,
                                 (x.size(4)-1)*self.stride-(self.kernel_size-1)*self.dilation+1)

        for down_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
            for right_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
                output_x = torch.zeros(x_res.size(0), x_res.size(1), 1,
                                            x_res.size(3), x_res.size(4))
                output_x[:, :, :, down_offset: x_res.size(3): self.stride,
                              right_offset: x_res.size(3): self.stride] = x
                x_res = torch.cat(x_res, output_x)

        return x_res[:, :, :, self.padding:-self.padding-1, self.padding:-self.padding-1]

    def deconv_kernel_channels_const(self, x, initialization_value):
        '''
        :param x: Input
        input to unroll, 4 dimensional
        :param initialization_value: default value for remaining elements
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        '''

        x = x.unsqueeze(2)
        x_res = torch.zeros(x.size(0), x.size(1), 0,
                                 (x.size(3)-1)*self.stride-(self.kernel_size-1)*self.dilation+1,
                                 (x.size(4)-1)*self.stride-(self.kernel_size-1)*self.dilation+1)

        for down_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
            for right_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
                output_x = initialization_value * torch.ones(x_res.size(0), x_res.size(1), 1,
                                                                  x_res.size(3), x_res.size(4))
                output_x[:, :, :, down_offset: x_res.size(3): self.stride, right_offset: x_res.size(3): self.stride] = x
                x_res = torch.cat(x_res, output_x)

        return x_res[:, :, :, self.padding:-self.padding-1, self.padding:-self.padding-1]
