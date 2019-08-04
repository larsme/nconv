
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

    def forward(self, d, cd, s, sc, gx, cgx, gy, cgy):

        ## get gradients

        gx_left = torch.roll(gx, shifts=(1), dims=(3))
        cgx_left = torch.roll(cgx, shifts=(1), dims=(3))
        # cgx_left[:, :, :, 0] = 0
        g_left = (cgx_left * gx_left + cgx*gx) / (cgx_left + cgx)
        cg_left = (cgx_left + cgx) / 2

        gx_right = torch.roll(gx, shifts=(-1), dims=(3))
        cgx_right = torch.roll(cgx, shifts=(-1), dims=(3))
        # cgx_right[:, :, :, -1] = 0
        g_right = -(cgx_right * gx_right + cgx*gx) / (cgx_right + cgx)
        cg_right = (cgx_right + cgx) / 2

        gy_up = torch.roll(gy, shifts=(1), dims=(2))
        cgy_up = torch.roll(cgy, shifts=(1), dims=(2))
        # cgy_up[:, :, 0, :] = 0
        g_up = (cgy_up * gy_up + cgy*gy) / (cgy_up + cgy)
        cg_up = (cgy_up + cgy) / 2

        gy_down = torch.roll(gy, shifts=(-1), dims=(2))
        cgy_down = torch.roll(cgy, shifts=(-1), dims=(2))
        # cgy_down[:, :, -1, :] = 0
        g_down = -(cgy_down * gy_down + cgy*gy) / (cgy_down + cgy)
        cg_down = (cgy_down + cgy) / 2

        gx_left_up = torch.roll(gx, shifts=(1, 1), dims=(3, 2))
        cgx_left_up = torch.roll(cgy, shifts=(1, 1), dims=(3, 2))
        # cgx_left_up[:, :, :, 0] = 0
        # cgx_left_up[:, :, 0, :] = 0
        gy_left_up = torch.roll(gy, shifts=(1, 1), dims=(3, 2))
        cgy_left_up = torch.roll(cgy, shifts=(1, 1), dims=(3, 2))
        # cgy_left_up[:, :, :, 0] = 0
        # cgy_left_up[:, :, 0, :] = 0
        g_left_up = (cgx_left_up * gx_left_up + cgx*gx) / (cgx_left_up + cgx) +  (cgy_left_up * gy_left_up + cgy*gy) / (cgy_left_up + cgy)
        cg_left_up = (cgx_left_up + cgx + cgy_left_up + cgy) / 4

        gx_right_up = torch.roll(gx, shifts=(-1, 1), dims=(3, 2))
        cgx_right_up = torch.roll(cgy, shifts=(-1, 1), dims=(3, 2))
        # cgx_right_up[:, :, :, -1] = 0
        # cgx_right_up[:, :, 0, :] = 0
        gy_right_up = torch.roll(gy, shifts=(-1, 1), dims=(3, 2))
        cgy_right_up = torch.roll(cgy, shifts=(-1, 1), dims=(3, 2))
        # cgy_right_up[:, :, :, -1] = 0
        # cgy_right_up[:, :, 0, :] = 0
        g_right_up = - (cgx_right_up * gx_right_up + cgx*gx) / (cgx_right_up + cgx) +  (cgy_right_up * gy_right_up + cgy*gy) / (cgy_right_up + cgy)
        cg_right_up = (cgx_right_up + cgx + cgy_right_up + cgy) / 4

        gx_left_down = torch.roll(gx, shifts=(1, -1), dims=(3, 2))
        cgx_left_down = torch.roll(cgy, shifts=(1, -1), dims=(3, 2))
        # cgx_left_down[:, :, :, 0] = 0
        # cgx_left_down[:, :, -1, :] = 0
        gy_left_down = torch.roll(gy, shifts=(1, -1), dims=(3, 2))
        cgy_left_down = torch.roll(cgy, shifts=(1, -1), dims=(3, 2))
        cgy_left_down[:, :, :, 0] = 0
        # cgy_left_down[:, :, -1, :] = 0
        g_left_down = (cgx_left_down * gx_left_down + cgx*gx) / (cgx_left_down + cgx) -  (cgy_left_down * gy_left_down + cgy*gy) / (cgy_left_down + cgy)
        cg_left_down = (cgx_left_down + cgx + cgy_left_down + cgy) / 4

        gx_right_down = torch.roll(gx, shifts=(-1, -1), dims=(3, 2))
        cgx_right_down = torch.roll(cgy, shifts=(-1, -1), dims=(3, 2))
        # cgx_right_down[:, :, :, -1] = 0
        # cgx_right_down[:, :, -1, :] = 0
        gy_right_down = torch.roll(gy, shifts=(-1, -1), dims=(3, 2))
        cgy_right_down = torch.roll(cgy, shifts=(-1, -1), dims=(3, 2))
        # cgy_right_down[:, :, :, -1] = 0
        # cgy_right_down[:, :, -1, :] = 0
        g_right_down = - (cgx_right_down * gx_right_down + cgx*gx) / (cgx_right_down + cgx) -  (cgy_right_down * gy_right_down + cgy*gy) / (cgy_right_down + cgy)
        cg_right_down = (cgx_right_down + cgx + cgy_right_down + cgy) / 4

        ## get d, dc

        d_left = torch.roll(d, shifts=(1), dims=(3)) * (1 + g_left)
        cd_left = torch.roll(cd * s, shifts=(1), dims=(3)) * s * (1+self.w_grad * cg_left) / (1+self.w_grad)
        cd_left[:, :, :, 0] = 0

        d_right = torch.roll(d, shifts=(-1), dims=(3)) * (1 + g_right)
        cd_right = torch.roll(cd * s, shifts=(-1), dims=(3)) * s * (1+self.w_grad * cg_right) / (1+self.w_grad)
        cd_right[:, :, :, -1] = 0

        d_up = torch.roll(d, shifts=(1), dims=(2)) * (1 + g_up)
        cd_up = torch.roll(cd * s, shifts=(1), dims=(2)) * s * (1+self.w_grad * cg_up) / (1+self.w_grad)
        cd_up[:, :, 0, :] = 0

        d_down = torch.roll(d, shifts=(-1), dims=(2)) * (1 + g_down)
        cd_down = torch.roll(cd * s, shifts=(-1), dims=(2)) * s * (1+self.w_grad * cg_down) / (1+self.w_grad)
        cd_down[:, :, -1, :] = 0

        d_left_up = torch.roll(d, shifts=(1, 1), dims=(3, 2)) * (1 + g_left_up)
        cd_left_up = torch.roll(cd * s, shifts=(1, 1), dims=(3, 2)) * s * (1+self.w_grad * cg_left_up) / (1+self.w_grad)
        cd_left_up[:, :, :, 0] = 0
        cd_left_up[:, :, 0, :] = 0

        d_right_up = torch.roll(d, shifts=(-1, 1), dims=(3, 2)) * (1 + g_right_up)
        cd_right_up = torch.roll(cd * s, shifts=(-1, 1), dims=(3, 2)) * s * (1+self.w_grad * cg_right_up) / (1+self.w_grad)
        cd_right_up[:, :, :, -1] = 0
        cd_right_up[:, :, 0, :] = 0

        d_left_down = torch.roll(d, shifts=(1, -1), dims=(3, 2)) * (1 + g_left_down)
        cd_left_down = torch.roll(cd * s, shifts=(1, -1), dims=(3, 2)) * s * (1+self.w_grad * cg_left_down) / (1+self.w_grad)
        cd_left_down[:, :, :, 0] = 0
        cd_left_down[:, :, -1, :] = 0

        d_right_down = torch.roll(d, shifts=(-1, -1), dims=(3, 2)) * (1 + g_right_down)
        cd_right_down = torch.roll(cd * s, shifts=(-1, -1), dims=(3, 2)) * s * (1+self.w_grad * cg_right_down) / (1+self.w_grad)
        cd_right_down[:, :, :, -1] = 0
        cd_right_down[:, :, -1, :] = 0

        # convolution
        

        # Normalized Convolution with border masking during propagation
        nomin = self.weight[0,0]* cd_left_up * d_left_up
        + self.weight[0,1]* cd_up * d_up
        + self.weight[0,2]* cd_right_up * d_right_up
        + self.weight[1,0]* cd_left * d_left
        + self.weight[1,1]* cd * d
        + self.weight[1,2]* cd_right * d_right
        + self.weight[2,0]* cd_left_down * d_left_down
        + self.weight[2,1]* cd_down * d_down
        + self.weight[2,2]* cd_right_down * d_right_down

        denomin = self.weight[0,0]* cd_left_up
        + self.weight[0,1]* cd_up
        + self.weight[0,2]* cd_right_up
        + self.weight[1,0]* cd_left
        + self.weight[1,1]* cd
        + self.weight[1,2]* cd_right
        + self.weight[2,0]* cd_left_down
        + self.weight[2,1]* cd_down
        + self.weight[2,2]* cd_right_down

        nconv = nomin / (denomin+self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denomin
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
            torch.nn.init.xavier_uniform_(self.w_grad)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
            torch.nn.init.kaiming_uniform_(self.w_grad)
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
