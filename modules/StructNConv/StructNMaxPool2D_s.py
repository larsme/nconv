
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


class StructNMaxPool2D_s(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, channels=5, pos_fn='softplus', init_method='p',
                 devalue_pooled_confidence=True):
        super(StructNMaxPool2D_s, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.devalue_pooled_confidence = devalue_pooled_confidence

        self.pos_fn = pos_fn
        self.init_method = init_method

        # Define Parameters
        self.w_s_pool = torch.nn.Parameter(data=torch.Tensor(1, channels, 1, 1))
        self.b_s_pool = torch.nn.Parameter(data=torch.Tensor(1, channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_pool)
            torch.nn.init.xavier_uniform_(self.b_s_pool)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_pool)
            torch.nn.init.kaiming_uniform_(self.b_s_pool)

    def forward(self, d, cd, s, cs, *args):
        _, inds = F.max_pool2d(cs*(self.w_s_pool*s + self.b_s_pool), kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)

        if self.devalue_pooled_confidence:
            return retrieve_indices(s, inds), retrieve_indices(cs, inds) / self.stride / self.stride
        else:
            return retrieve_indices(s, inds), retrieve_indices(cs, inds)
