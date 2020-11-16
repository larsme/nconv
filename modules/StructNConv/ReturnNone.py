
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch


class ReturnNone(torch.nn.Module):
    def __init__(self):
        super(ReturnNone, self).__init__()
        
    def enforce_limits(self):
        return

    def forward(self, *args):
        return None

