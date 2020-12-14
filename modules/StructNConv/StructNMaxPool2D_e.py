
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


class StructNMaxPool2D_e(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, channels=5, devalue_pooled_confidence=True):
        super(StructNMaxPool2D_e, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1
                        
    def enforce_limits(self):
        return

    def forward(self, d, cd, e, ce, *args):
        # edge directions are converted to channels, actual pooling is the same as in StructNMaxPool2D_s
        # this means the new edge directions may originate from different locations in the pooling field
        
        #edges:
        #0 => /
        #1 => -
        #2 => \
        #3 => |
        e, ce = e.view(e.shape[0], e.shape[1] * e.shape[2], e.shape[3], e.shape[4]), ce.view(e.shape[0], e.shape[1] * e.shape[2], e.shape[3], e.shape[4])
        _, inds = F.max_pool2d(ce * (1 - e), kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)
        shape = inds.shape

        return retrieve_indices(e, inds).view(shape[0], shape[1] // 4, 4, shape[2], shape[3]), retrieve_indices(ce, inds).view(shape[0], shape[1] // 4, 4, shape[2], shape[3]) * self.devalue_conf
