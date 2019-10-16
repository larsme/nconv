
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F

from modules.NConv2D import EnforcePos


class SobelEdge(torch.nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()

        # Define Parameters
        self.sobel_weight_1 = torch.nn.Parameter(data=torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                                                 .unsqueeze(0).unsqueeze(0),
                                                 requires_grad=False)
        self.sobel_weight_2 = torch.nn.Parameter(data=torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                                                 .unsqueeze(0).unsqueeze(0),
                                                 requires_grad=False)
        self.channel_weights = torch.nn.Parameter(data=torch.Tensor(1, 3, 1, 1))

    def forward(self, rgb):
        single_channel = F.pad(F.conv2d(rgb, self.channel_weights), (1, 1, 1, 1), mode='replicate')
        s = torch.exp(torch.abs(F.conv2d(single_channel, self.sobel_weight_1)))\
            * torch.exp(torch.abs(F.conv2d(single_channel, self.sobel_weight_2)))
        cs = torch.ones_like(s)

        return s, cs
