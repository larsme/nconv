
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch.nn.functional as F
import torch.nn as nn
from modules.StructNConv.retrieve_indices import retrieve_indices

class StructNMaxPool2D_d_with_s(nn.modules.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(StructNMaxPool2D_d_with_s, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, d, cd, s, cs):
        _, inds = F.max_pool2d(cd*s, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)
        return retrieve_indices(d, inds), retrieve_indices(cd, inds) / self.stride / self.stride
