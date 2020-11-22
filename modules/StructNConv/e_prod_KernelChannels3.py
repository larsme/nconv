
import torch
import torch.nn.functional as F
from modules.StructNConv.KernelChannels import KernelChannels


class e_prod_KernelChannels3(torch.nn.Module):
    def __init__(self):
        super(e_prod_KernelChannels3, self).__init__()

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

        edges:
        0 => /
        1 => -
        2 => \
        3 => |
        '''
        s_ext = F.pad(s, (1,1,1,1))
        return torch.stack((s[:,:,0,:,:].clone() * s_ext[:,:,0,:-2,:-2],  s[:,:,1,:,:].clone()*s_ext[:,:,1,:-2,1:-1], s[:,:,2,:,:].clone()*s_ext[:,:,2,:-2,2:],
                            s[:,:,3,:,:].clone() * s_ext[:,:,3,1:-1,:-2], torch.ones_like(s[:,:,1,:,:]),              s[:,:,3,:,:].clone()*s_ext[:,:,3,1:-1,2:],
                            s[:,:,2,:,:].clone() * s_ext[:,:,2,2:,:-2],   s[:,:,1,:,:].clone()*s_ext[:,:,1,2:,1:-1],  s[:,:,0,:,:].clone()*s_ext[:,:,0,2:,2:]), dim=2)

