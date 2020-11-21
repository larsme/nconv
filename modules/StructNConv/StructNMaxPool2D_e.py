
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
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, channels=5, init_method='p',
                 devalue_pooled_confidence=True):
        super(StructNMaxPool2D_e, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1

        self.init_method = init_method

        # Define Parameters
        self.w_s_pool = torch.nn.Parameter(data=torch.Tensor(1, channels * 4, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_pool) + 1
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_pool)
                        
    def enforce_limits(self):
        # Enforce weights between -1 and 1
        self.w_s_pool.data = torch.tanh(self.w_s_pool)

    def forward(self, d, cd, s, cs, *args):
        
        #edges:
        #0 => /
        #1 => -
        #2 => \
        #3 => |
        s, cs = s.view(s.shape[0], s.shape[1] * s.shape[2], s.shape[3], s.shape[4]), cs.view(s.shape[0], s.shape[1] * s.shape[2], s.shape[3], s.shape[4])
        _, inds = F.max_pool2d(cs * (self.w_s_pool * s + 1), kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, return_indices=True)
        shape=inds.shape

        return retrieve_indices(s, inds).view(shape[0], shape[1] // 4, 4, shape[2], shape[3]), retrieve_indices(cs, inds).view(shape[0], shape[1] // 4, 4, shape[2], shape[3]) * self.devalue_conf
