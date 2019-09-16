
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
from modules.NConv2D import EnforcePos


class StructNMaxPool2D_dg_with_s(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, channels=5, pos_fn='softplus', init_method='p',
                 devalue_pooled_confidence=True):
        super(StructNMaxPool2D_dg_with_s, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.devalue_pooled_confidence = devalue_pooled_confidence

        self.pos_fn = pos_fn
        self.init_method = init_method

        # Define Parameters
        self.wg = torch.nn.Parameter(data=torch.Tensor(1, channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.wg)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.wg)

        # Enforce positive weights
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'wg', pos_fn)

    def forward(self, d, cd, s, cs, gx, cgx, gy, cgy, *args):
        _, inds = F.max_pool2d(s * cd * (cgx + cgy + self.wg), kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)

        if self.devalue_pooled_confidence:
            return retrieve_indices(d, inds), retrieve_indices(cd, inds) / self.stride / self.stride, \
                retrieve_indices(gx, inds)*self.stride, retrieve_indices(cgx, inds) / self.stride / self.stride, \
                retrieve_indices(gy, inds)*self.stride, retrieve_indices(cgy, inds) / self.stride / self.stride
        else:
            return retrieve_indices(d, inds), retrieve_indices(cd, inds), \
                retrieve_indices(gx, inds)*self.stride, retrieve_indices(cgx, inds), \
                retrieve_indices(gy, inds)*self.stride, retrieve_indices(cgy, inds)
