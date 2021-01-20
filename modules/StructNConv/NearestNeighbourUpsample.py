
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F


class NearestNeighbourUpsample(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(NearestNeighbourUpsample, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def prep_eval(self):
        return

    def forward(self, d, cd):
        output_size = ((d.size(2)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1,
                       (d.size(3)-1)*self.stride-2*self.padding+(self.kernel_size-1)*self.dilation+1)
        d = F.interpolate(d, output_size, mode='nearest')
        cd = F.interpolate(cd, output_size, mode='nearest')
        return d, cd
    
