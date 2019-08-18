
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

    def kernel_channels(self, tensor):
        '''
        :param tensor: Input
        input to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6], 
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        '''

        tensor = F.pad(tensor, (self.padding, self.padding, self.padding, self.padding)).unsqueeze(2)
        tensor_res = torch.zeros(tensor.size(0), tensor.size(1), 0,
                                 (tensor.size(3)-(self.dilation*self.kernel_size-1)-1)/self.stride+1,
                                 (tensor.size(4)-(self.dilation*self.kernel_size-1)-1)/self.stride+1)

        for down_offset in range(0, self.kernel_size*self.dilation, self.dilation):
            for right_offset in range(0, self.kernel_size*self.dilation, self.dilation):
                tensor_res = torch.cat(tensor_res, tensor[:, :, :,
                                                          down_offset : self.stride
                                                          : tensor_res.size(3) * self.stride + down_offset,
                                                          right_offset : self.stride
                                                          : tensor_res.size(3) * self.stride + right_offset])

        return tensor_res

    def deconv_kernel_channels(self, tensor):
        '''
        :param tensor: Input
        input to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        '''

        tensor = tensor.unsqueeze(2)
        tensor_res = torch.zeros(tensor.size(0), tensor.size(1), 0,
                                 (tensor.size(3)-1)*self.stride-(self.kernel_size-1)*self.dilation+1,
                                 (tensor.size(4)-1)*self.stride-(self.kernel_size-1)*self.dilation+1)

        for down_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
            for right_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
                output_tensor = torch.zeros(tensor_res.size(0), tensor_res.size(1), 1,
                                            tensor_res.size(3), tensor_res.size(4))
                output_tensor[:, :, :, down_offset : self.stride : tensor_res.size(3),
                              right_offset : self.stride : tensor_res.size(3)] = tensor
                tensor_res = torch.cat(tensor_res, output_tensor)

        return tensor_res[:, :, :, self.padding:-self.padding-1, self.padding:-self.padding-1]

    def deconv_kernel_channels_const(self, tensor, initialization_value):
        '''
        :param tensor: Input
        input to unroll, 4 dimensional
        :param initialization_value: default value for remaining elements
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        '''

        tensor = tensor.unsqueeze(2)
        tensor_res = torch.zeros(tensor.size(0), tensor.size(1), 0,
                                 (tensor.size(3)-1)*self.stride-(self.kernel_size-1)*self.dilation+1,
                                 (tensor.size(4)-1)*self.stride-(self.kernel_size-1)*self.dilation+1)

        for down_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
            for right_offset in range(self.kernel_size*self.dilation-1, 0, -self.dilation):
                output_tensor = initialization_value * torch.ones(tensor_res.size(0), tensor_res.size(1), 1,
                                                                  tensor_res.size(3), tensor_res.size(4))
                output_tensor[:, :, :, down_offset : self.stride : tensor_res.size(3),
                              right_offset : self.stride : tensor_res.size(3)] = tensor
                tensor_res = torch.cat(tensor_res, output_tensor)

        return tensor_res[:, :, :, self.padding:-self.padding-1, self.padding:-self.padding-1]
