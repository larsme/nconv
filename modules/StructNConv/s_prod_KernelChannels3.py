
import torch
from modules.StructNConv.KernelChannels import KernelChannels


class s_prod_KernelChannels3(torch.nn.Module):
    def __init__(self):
        super(s_prod_KernelChannels3, self).__init__()
        self.kernel_channels = KernelChannels(3, 1, 1, 1)

    def forward(self, s, cs):
        '''
        :param
        s, cs to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        s of middle = 1, cs of middle = 1
        cs_out = product of cs along path to middle (including middle) with highest cs
        s_out = product of s along same path
        '''
        s_roll = self.kernel_channels.kernel_channels(s)*s [:,:,None,:,:]       
        s_roll[:,:,4,:,:]=1

        return s_roll

