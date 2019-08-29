
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F

from modules.StructNConv.retrieve_indices import retrieve_indices


class StructNMaxPool2D_d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(StructNMaxPool2D_d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, d, cd):
        best_cd, inds = F.max_pool2d(cd, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, return_indices=True)
        return retrieve_indices(d, inds), best_cd / self.stride / self.stride
