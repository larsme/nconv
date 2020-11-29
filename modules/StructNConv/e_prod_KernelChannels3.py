
import torch
import torch.nn.functional as F
from modules.StructNConv.KernelChannels import KernelChannels


class e_prod_KernelChannels3(torch.nn.Module):
    def __init__(self, stride=1):
        super(e_prod_KernelChannels3, self).__init__()
        self.stride = stride

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
        return torch.stack((s[:,:,0,::self.stride,::self.stride].clone() * s_ext[:,:,0,:-2:self.stride,:-2:self.stride],  s[:,:,1,::self.stride,::self.stride].clone() * s_ext[:,:,1,:-2:self.stride,1:-1:self.stride], s[:,:,2,::self.stride,::self.stride].clone() * s_ext[:,:,2,:-2:self.stride,2::self.stride],
                            s[:,:,3,::self.stride,::self.stride].clone() * s_ext[:,:,3,1:-1:self.stride,:-2:self.stride], torch.ones_like(s[:,:,1,::self.stride,::self.stride]),                                        s[:,:,3,::self.stride,::self.stride].clone() * s_ext[:,:,3,1:-1:self.stride,2::self.stride],
                            s[:,:,2,::self.stride,::self.stride].clone() * s_ext[:,:,2,2::self.stride,:-2:self.stride],   s[:,:,1,::self.stride,::self.stride].clone() * s_ext[:,:,1,2::self.stride,1:-1:self.stride],  s[:,:,0,::self.stride,::self.stride].clone() * s_ext[:,:,0,2::self.stride,2::self.stride]), dim=2)

