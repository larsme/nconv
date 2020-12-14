
import torch
import torch.nn.functional as F
from modules.StructNConv.KernelChannels import KernelChannels


class e_prod_KernelChannels3(torch.nn.Module):
    def __init__(self, stride=1):
        super(e_prod_KernelChannels3, self).__init__()
        self.stride = stride

    def forward(self, e, ce):
        '''
        :param
        e, ce to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        e of middle = 1, ce of middle = 1
        cs_out = product of ce along path to middle (including middle) with highest ce
        s_out = product of e along same path

        edges:
        0 => /
        1 => -
        2 => \
        3 => |
        '''
        e_ext = F.pad(e, (1,1,1,1))
        return torch.stack((e[:,:,0,::self.stride,::self.stride].clone() * e_ext[:,:,0,:-2:self.stride,:-2:self.stride],  e[:,:,1,::self.stride,::self.stride].clone() * e_ext[:,:,1,:-2:self.stride,1:-1:self.stride], e[:,:,2,::self.stride,::self.stride].clone() * e_ext[:,:,2,:-2:self.stride,2::self.stride],
                            e[:,:,3,::self.stride,::self.stride].clone() * e_ext[:,:,3,1:-1:self.stride,:-2:self.stride], torch.ones_like(e[:,:,1,::self.stride,::self.stride]),                                        e[:,:,3,::self.stride,::self.stride].clone() * e_ext[:,:,3,1:-1:self.stride,2::self.stride],
                            e[:,:,2,::self.stride,::self.stride].clone() * e_ext[:,:,2,2::self.stride,:-2:self.stride],   e[:,:,1,::self.stride,::self.stride].clone() * e_ext[:,:,1,2::self.stride,1:-1:self.stride],  e[:,:,0,::self.stride,::self.stride].clone() * e_ext[:,:,0,2::self.stride,2::self.stride]), dim=2)

