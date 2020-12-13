
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
    def __init__(self, kernel_size, stride=1, padding=0, channels=5, devalue_pooled_confidence=True):
        super(StructNMaxPool2D_e, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1
        self.eps = 1e-20
                        
    def enforce_limits(self):
        return

    def forward(self, d, cd, s, cs, *args):
        # edge directions are converted to channels, actual pooling is the same as in StructNMaxPool2D_s
        # this means the new edge directions may originate from different locations in the pooling field
        
        #edges:
        #0 => /
        #1 => -
        #2 => \
        #3 => |
        s, cs = s.view(s.shape[0], s.shape[1] * s.shape[2], s.shape[3], s.shape[4]), cs.view(s.shape[0], s.shape[1] * s.shape[2], s.shape[3], s.shape[4])
        w = cs * (1 - s)
        nom_s = F.avg_pool2d(w * s, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        nom_cs = F.avg_pool2d(w * cs, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        denom = F.avg_pool2d(w, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) + self.eps
        shape = nom_s.shape
        
        return (nom_s / denom).view(shape[0], shape[1]//4,4,shape[2], shape[3]), (nom_cs / denom).view(shape[0], shape[1]//4,4,shape[2], shape[3])