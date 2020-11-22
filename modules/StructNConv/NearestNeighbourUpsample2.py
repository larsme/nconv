
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F


class NearestNeighbourUpsample2(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(NearestNeighbourUpsample2, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def enforce_limits(self):
        return

    def forward(self, d, cd):
        # dims 1 and 2 are combined, rest is like the normal version
        shape=d.shape
        d,cd = d.view(shape[0], shape[1]*shape[2], shape[3], shape[4]), cd.view(shape[0], shape[1]*shape[2], shape[3], shape[4])
        output_size = ((d.size(2)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1,
                       (d.size(3)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1)
        d = F.interpolate(d, output_size, mode='nearest')
        cd = F.interpolate(cd, output_size, mode='nearest')
        shape2=d.shape
        return d.view(shape[0], shape[1],shape[2], shape2[2], shape2[3]), cd.view(shape[0], shape[1],shape[2], shape2[2], shape2[3])
    
