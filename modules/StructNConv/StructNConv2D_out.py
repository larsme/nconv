
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch
import torch.nn.functional as F


class StructNConv2D_out(torch.nn.Module):
    def __init__(self, init_method='k', in_channels=1):
        super(StructNConv2D_out, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method

        self.in_channels = in_channels

        # Define Parameters
        if self.in_channels>1:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 1, 1))
            # Init Parameters
            if self.init_method == 'x':  # Xavier
                torch.nn.init.xavier_uniform_(self.channel_weight)+1
            else:  # elif self.init_method == 'k': # Kaiming
                torch.nn.init.kaiming_uniform_(self.channel_weight)
        #self.unseen = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))

    def print(self, s_list):
        return s_list
            
    def enforce_limits(self):
        # Enforce positive weights
        if self.in_channels>1:
            self.channel_weight.data = F.softplus(self.channel_weight, beta=10)

    def forward(self, d, cd):
                
        if self.in_channels>1:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cd * d, self.channel_weight)
            denom = F.conv2d(cd, self.channel_weight)
            d = nom / (denom + self.eps)
            cd = denom / (torch.sum(self.channel_weight) + self.eps)

        #d[cd == 0] = self.unseen
        return d, cd
