
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal

from modules.NConv2D import EnforcePos
from modules.StructNConv.KernelChannels import KernelChannels
from modules.StructNConv.retrieve_indices import retrieve_indices


class s_prod_KernelChannels(nn.modules.Module):
    def __init__(self,  kernel_size, stride=1, padding=0, dilation=1):
        super(s_prod_KernelChannels, self).__init__()
        self.kernel_size = kernel_size
        self.dilated_kernel_size = 1+(kernel_size-1)*dilation
        self.stride = stride
        self.dilation = dilation
        self.kernel_channels = KernelChannels(1+(kernel_size-1)*dilation, 1, padding, 1)

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
        s_roll = self.kernel_channels.kernel_channels(s)
        cs_roll = self.kernel_channels.kernel_channels(cs)
        s_out = torch.ones_like(s_roll)
        cs_out = torch.ones_like(s_roll)

        mid = int((self.dilated_kernel_size-1)/2)*self.dilated_kernel_size+int((self.dilated_kernel_size-1)/2)

        if self.dilated_kernel_size > 1:
            c = self.dilated_kernel_size + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = self.dilated_kernel_size + 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]

        if self.dilated_kernel_size > 2:
            c = -self.dilated_kernel_size + 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -self.dilated_kernel_size + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -self.dilated_kernel_size - 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            c = +self.dilated_kernel_size - 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]

        for d in range(2, int((self.dilated_kernel_size+1)/2)):

            #corners, next to corners
            # upper left
            corner = -d * self.dilated_kernel_size - d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_roll[:, :, corner+self.dilated_kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_roll[:, :, corner+self.dilated_kernel_size+1, :, :]

            c = corner + self.dilated_kernel_size
            s_candidates = s_out[:, :, [c+1, c+self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c+self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            # upper right
            corner = -d * self.dilated_kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_roll[:, :, corner+self.dilated_kernel_size-1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :] \
                                         * cs_roll[:, :, corner+self.dilated_kernel_size-1, :, :]

            c = corner + self.dilated_kernel_size
            s_candidates = s_out[:, :, [c-1, c+self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, c+self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            # lower left
            corner = +d * self.dilated_kernel_size - d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_roll[:, :, corner-self.dilated_kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_roll[:, :, corner-self.dilated_kernel_size+1, :, :]

            c = corner - self.dilated_kernel_size
            s_candidates = s_out[:, :, [c+1, c-self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c-self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            # lower right
            corner = +d * self.dilated_kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_roll[:, :, corner-self.dilated_kernel_size-1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_roll[:, :, corner-self.dilated_kernel_size-1, :, :]

            c = corner - self.dilated_kernel_size
            s_candidates = s_out[:, :, [c-1, c-self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, -self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]


            # side middles
            # left
            x = -d
            for y in range(-d-2, +d-1):
                c = y*self.dilated_kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size+1,
                                            c+1,
                                            c+self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size+1,
                                              c+1,
                                              c+self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            #top
            y = -d
            for x in range(-d-2, +d-1):
                c = y*self.dilated_kernel_size+x+mid
                s_candidates = s_out[:, :, [c+self.dilated_kernel_size-1,
                                            c+self.dilated_kernel_size,
                                            c+self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size-1,
                                              c+self.dilated_kernel_size,
                                              c+self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]
            # right
            x = +d
            for y in range(-d-2, +d-1):
                c = y*self.dilated_kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size-1,
                                            c-1,
                                            c+self.dilated_kernel_size-1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size-1,
                                              c-1,
                                              c+self.dilated_kernel_size-1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]
            # bottom
            y = +d
            for x in range(-d-2, +d-1):
                c = y*self.dilated_kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size-1,
                                            c-self.dilated_kernel_size,
                                            c-self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size-1,
                                              c-self.dilated_kernel_size,
                                              c-self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

        if self.dilated_kernel_size % 2 == 0 and self.dilated_kernel_size > 3:
            # extra bot, right
            # corners
            d = int(self.dilated_kernel_size/2)

            # upper right
            corner = -d * self.dilated_kernel_size + d + mid
            c = corner + self.dilated_kernel_size
            s_candidates = s_out[:, :, [c - 1,
                                        c + self.dilated_kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - 1,
                                          c + self.dilated_kernel_size - 1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]


            # lower left
            corner = +d * self.dilated_kernel_size - d + mid
            c = corner + 1
            s_candidates = s_out[:, :, [c - self.dilated_kernel_size,
                                        c - self.dilated_kernel_size + 1], :, :]
            cs_candidates = cs_out[:, :, [c - self.dilated_kernel_size,
                                          c - self.dilated_kernel_size + 1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            # lower right
            corner = +d * self.dilated_kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :] \
                                        * s_roll[:, :, corner - self.dilated_kernel_size - 1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :] \
                                         * cs_roll[:, :, corner - self.dilated_kernel_size - 1, :, :]

            c = corner - self.dilated_kernel_size
            s_candidates = s_out[:, :, [c - 1,
                                        c - self.dilated_kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - 1,
                                          c -self.dilated_kernel_size - 1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c - self.dilated_kernel_size,
                                        c - self.dilated_kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - self.dilated_kernel_size,
                                          c - self.dilated_kernel_size - 1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

            # side middles
            # right
            x = +d
            for y in range(-d - 2, +d - 1):
                c = y * self.dilated_kernel_size + x + mid
                s_candidates = s_out[:, :, [c - self.dilated_kernel_size - 1,
                                            c - 1,
                                            c + self.dilated_kernel_size - 1], :, :]
                cs_candidates = cs_out[:, :, [c - self.dilated_kernel_size - 1,
                                              c - 1,
                                              c + self.dilated_kernel_size - 1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]
            # bottom
            y = +d
            for x in range(-d - 2, +d - 1):
                c = y * self.dilated_kernel_size + x + mid
                s_candidates = s_out[:, :, [c - self.dilated_kernel_size - 1,
                                            c - self.dilated_kernel_size,
                                            c - self.dilated_kernel_size + 1], :, :]
                cs_candidates = cs_out[:, :, [c - self.dilated_kernel_size - 1,
                                              c - self.dilated_kernel_size,
                                              c - self.dilated_kernel_size + 1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_out[:, :, c, :, :]

        dilation_items = torch.arange(0, self.dilated_kernel_size, self.dilation)
        dilation_items = dilation_items.unsqueeze(0).expand(self.kernel_size, -1).flatten()\
                         + dilation_items.unsqueeze(1).expand(-1, self.kernel_size).flatten()*self.dilated_kernel_size
        s_out = s_out[:, :, dilation_items,
                      : s_out.shape[3]: self.stride, : s_out.shape[4]: self.stride]
        return s_out
