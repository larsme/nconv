
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


class s_prod_KernelChannels_deconv(KernelChannels):
    def __init__(self,  kernel_size, stride=1, padding=0, dilation=1, pos_fn='softplus', init_method='k'):
        super(KernelChannels, self).__init__(kernel_size, stride, padding, dilation)

        self.pos_fn = pos_fn

        # Define Parameters
        self.w_unknown_s = torch.nn.Parameter(data=torch.Tensor(1))
        self.w_unknown_cs = torch.nn.Parameter(data=torch.Tensor(1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_unknown_s)
            torch.nn.init.xavier_uniform_(self.w_unknown_cs)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_unknown_s)
            torch.nn.init.kaiming_uniform_(self.w_unknown_cs)

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'w_unknown_s', pos_fn)
            EnforcePos.apply(self, 'w_unknown_cs', pos_fn)

    def s_prod_deconv_kernel_channels(self, s, cs):
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
        stride = self.stride
        dilation = self.dilation
        kernel_size = self.kernel_size

        self.stride=0
        self.dilation = 1
        self.kernel_size=1+(kernel_size-1)*dilation

        s_roll = self.deconv_kernel_channels_const(s, self.w_unknown_s)
        cs_roll = self.deconv_kernel_channels_const(cs, self.w_unknown_cs)
        deconv_present = self.deconv_kernel_channels(torch.ones_like(s))
        s_out = torch.ones_like(s_roll)
        cs_out = torch.ones_like(s_roll)

        mid = (self.kernel_size/2)*self.kernel_size+self.kernel_size/2

        if self.kernel_size > 1:
            c = self.kernel_size + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = self.kernel_size + 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]

        if self.kernel_size > 2:
            c = -self.kernel_size + 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -self.kernel_size + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -self.kernel_size - 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
            c = -1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            c = +self.kernel_size - 1 + mid
            s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]

        for d in range(2, (self.kernel_size-1)/2):
            # corners, next to corners
            # upper left
            corner = -d * self.kernel_size - d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]*s_roll[:, :, corner+self.kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]*cs_roll[:, :, corner+self.kernel_size+1, :, :]

            c = corner + self.kernel_size
            s_candidates = s_out[:, :, [c+1, c+self.kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c+self.kernel_size+1], :, :]
            candidates_present = deconv_present[:, :, [c+1, c+self.kernel_size+1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c+self.kernel_size, c+self.kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+self.kernel_size, c+self.kernel_size+1], :, :]
            candidates_present = deconv_present[:, :, [c+self.kernel_size, c+self.kernel_size+1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # upper right
            corner = - d * self.kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]*s_roll[:, :, corner+self.kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]*cs_roll[:, :, corner+self.kernel_size+1, :, :]

            c = corner + self.kernel_size
            s_candidates = s_out[:, :, [c-1, c+self.kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, c+self.kernel_size-1], :, :]
            candidates_present = deconv_present[:, :, [c-1, c+self.kernel_size-1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c+self.kernel_size, c+self.kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c+self.kernel_size, c+self.kernel_size-1], :, :]
            candidates_present = deconv_present[:, :, [c+self.kernel_size, c+self.kernel_size-1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # lower left
            corner = + d * self.kernel_size - d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]*s_roll[:, :, corner-self.kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]*cs_roll[:, :, corner-self.kernel_size+1, :, :]

            c = corner - self.kernel_size
            s_candidates = s_out[:, :, [c+1, c-self.kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c-self.kernel_size+1], :, :]
            candidates_present = deconv_present[:, :, [c+1, c-self.kernel_size+1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c-self.kernel_size, c-self.kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c-self.kernel_size, c-self.kernel_size+1], :, :]
            candidates_present = deconv_present[:, :, [c-self.kernel_size, c-self.kernel_size+1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # lower right
            corner = + d * self.kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]*s_roll[:, :, corner-self.kernel_size+1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]*cs_roll[:, :, corner-self.kernel_size+1, :, :]

            c = corner - self.kernel_size
            s_candidates = s_out[:, :, [c-1, c-self.kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, -self.kernel_size-1], :, :]
            candidates_present = deconv_present[:, :, [c-1, -self.kernel_size-1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c-self.kernel_size, c-self.kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-self.kernel_size, c-self.kernel_size-1], :, :]
            candidates_present = deconv_present[:, :, [c-self.kernel_size, c-self.kernel_size-1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # side middles
            # left
            x = -d
            for y in range(mid-d-2, mid+d-1):
                c = y*self.kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.kernel_size+1, c+1, c+self.kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.kernel_size+1, c+1, c+self.kernel_size+1], :, :]
                candidates_present = deconv_present[:, :, [c-self.kernel_size+1, c+1, c+self.kernel_size+1], :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # top
            y = -d
            for x in range(mid-d-2, mid+d-1):
                c = y*self.kernel_size+x+mid
                s_candidates = s_out[:, :, [c+self.kernel_size-1, c+self.kernel_size, c+self.kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c+self.kernel_size-1, c+self.kernel_size, c+self.kernel_size+1], :, :]
                candidates_present = deconv_present[:, :,
                                                    [c+self.kernel_size-1, c+self.kernel_size, c+self.kernel_size+1],
                                                    :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]
            # right
            x = +d
            for y in range(mid-d-2, mid+d-1):
                c = y*self.kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.kernel_size-1, c-1, c+self.kernel_size-1], :, :]
                cs_candidates = cs_out[:, :, [c-self.kernel_size-1, c-1, c+self.kernel_size-1], :, :]
                candidates_present = deconv_present[:, :,
                                                    [c - self.kernel_size - 1, c - 1, c + self.kernel_size - 1], :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]
            # bottom
            y = +d
            for x in range(mid-d-2, mid+d-1):
                c = y*self.kernel_size+x+mid
                s_candidates = s_out[:, :, [c-self.kernel_size-1, c-self.kernel_size, c-self.kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.kernel_size-1, c-self.kernel_size, c-self.kernel_size+1], :, :]
                candidates_present = deconv_present[:, :,
                                                    [c-self.kernel_size-1, c-self.kernel_size, c-self.kernel_size+1],
                                                    :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

        if self.kernel_size % 2 == 0 and self.kernel_size > 3:
            d = self.kernel_size/2
            # extra bot, right

            # corners

            # upper right
            corner = -d * self.kernel_size + d + mid
            c = corner + self.kernel_size
            s_candidates = s_out[:, :, [c - 1, c + self.kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - 1, c + self.kernel_size - 1], :, :]
            candidates_present = deconv_present[:, :, [c - 1, c + self.kernel_size - 1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # lower left
            corner = +d * self.kernel_size - d + mid
            c = corner + 1
            s_candidates = s_out[:, :, [c - self.kernel_size, c - self.kernel_size + 1], :, :]
            cs_candidates = cs_out[:, :, [c - self.kernel_size, c - self.kernel_size + 1], :, :]
            candidates_present = deconv_present[:, :, [c - self.kernel_size, c - self.kernel_size + 1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # lower right
            corner = +d * self.kernel_size + d + mid
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :] * s_roll[:, :, corner - self.kernel_size + 1, :, :]
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :] * cs_roll[:, :, corner - self.kernel_size + 1, :,
                                                                               :]

            c = corner - self.kernel_size
            s_candidates = s_out[:, :, [c - 1, c - self.kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - 1, -self.kernel_size - 1], :, :]
            candidates_present = deconv_present[:, :, [c - 1, -self.kernel_size - 1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c - self.kernel_size, c - self.kernel_size - 1], :, :]
            cs_candidates = cs_out[:, :, [c - self.kernel_size, c - self.kernel_size - 1], :, :]
            candidates_present = deconv_present[:, :, [c - self.kernel_size, c - self.kernel_size - 1], :, :]
            _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (2, 1, 1),
                                                          return_indices=True)
            s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
            cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

            # side middles
            # right
            x = +d
            for y in range(mid - d - 2, mid + d - 1):
                c = y * self.kernel_size + x + mid
                s_candidates = s_out[:, :, [c - self.kernel_size - 1, c - 1, c + self.kernel_size - 1], :, :]
                cs_candidates = cs_out[:, :, [c - self.kernel_size - 1, c - 1, c + self.kernel_size - 1], :, :]
                candidates_present = deconv_present[:, :,
                                                    [c - self.kernel_size - 1, c - 1, c + self.kernel_size - 1],
                                                    :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]
            # bottom
            y = +d
            for x in range(mid - d - 2, mid + d - 1):
                c = y * self.kernel_size + x + mid
                s_candidates = s_out[:, :, [c - self.kernel_size - 1, c - self.kernel_size, c - self.kernel_size + 1],
                                     :, :]
                cs_candidates = cs_out[:, :, [c - self.kernel_size - 1, c - self.kernel_size, c - self.kernel_size + 1],
                                       :, :]
                candidates_present = deconv_present[:, :, [c - self.kernel_size - 1,
                                                           c - self.kernel_size, c - self.kernel_size + 1], :, :]
                _, best_inds = torch.nn.functional.max_pool3d(cs_candidates * candidates_present, (3, 1, 1),
                                                              return_indices=True)
                s_out[:, :, c, :, :] = s_candidates[best_inds] * s_out[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_candidates[best_inds] * cs_out[:, :, c, :, :]

        self.stride = stride
        s_out = s_out[:, :, :, :self.stride:, :self.stride:]

        dilation_items = torch.arange(0, dilation, self.kernel_size)
        y_steps, x_steps = torch.meshgrid(dilation_items, dilation_items)
        s_out = s_out[:, :, y_steps*kernel_size+x_steps, :, :]
        self.dilation = dilation
        self.kernel_size = kernel_size

        return s_out



