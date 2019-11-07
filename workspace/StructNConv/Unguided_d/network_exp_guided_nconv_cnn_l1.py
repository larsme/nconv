########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import importlib
import sys
import os as os


class CNN(nn.Module):

    def __init__(self, params):
        super().__init__()

        # Import the unguided network
        unguided_module_file = 'workspace/StructNConv/Unguided_d/Unguided_d'
        f = importlib.import_module(unguided_module_file.replace('/', '.'))
        self.d_net = f.CNN(params)
        checkpoint_dict = torch.load('workspace/StructNConv/Unguided_d/ExperimentsNewDataset/'
                                     'Xavier_no_bias_lidar_padding_rotate_90/1/checkpoints/CNN_ep0010.pth.tar')
        self.d_net.load_state_dict(checkpoint_dict['net'])
        
        # Disable Training for the unguided module
        for p in self.d_net.parameters():            
            p.requires_grad=False
        
        self.d = nn.Sequential(
          nn.Conv2d(1,4,3,1,1),
          nn.ReLU(),
          nn.Conv2d(4,8,3,1,1),
          nn.ReLU(),                                              
        )#11,664 Params
        
        # RGB stream
        self.rgb = nn.Sequential(
          nn.Conv2d(4,8,3,1,1),
          nn.ReLU(),
          nn.Conv2d(8,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3,1,1),
          nn.ReLU(),                                            
        )

        # Fusion stream
        self.fuse = nn.Sequential(
          nn.Conv2d(40,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,4,3,1,1),
          nn.ReLU(),
          nn.Conv2d(4,1,1,1),
        )
            
        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

                
            
    def forward(self, x0_d, c0, x0_rgb ):  
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)
        
        xout_d = self.d(xout_d)
        
        self.xout_d = xout_d
        self.cout_d = cout_d
                

        # RGB network
        xout_rgb = self.rgb(torch.cat((x0_rgb, cout_d),1))
        self.xout_rgb = xout_rgb
        
        # Fusion Network
        xout = self.fuse(torch.cat((xout_rgb, xout_d),1))
        
        return xout, cout_d
       
