
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F


class StructNConv2D_d_with_e(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, in_channels=1, out_channels=1, groups=1,
                 kernel_size=1, stride=1, padding=0, devalue_pooled_confidence=True):
        super(StructNConv2D_d_with_e, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.mirror_weights = mirror_weights
        
        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1

        # Define Parameters
        if mirror_weights:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, (self.kernel_size - 1) // 2, 3, 2))
        else:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, (self.kernel_size - 1) // 2, 3, 3))
        if self.in_channels > 1:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 1, 1))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            if self.in_channels > 1:
                torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(spatial_weight)
        else:  # elif self.init_method == 'k': # Kaiming
            if self.in_channels > 1:
                torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(spatial_weight)
        
        if mirror_weights:
            self.true_spatial_weight = spatial_weight
        else:
            self.spatial_weight = spatial_weight.view(self.out_channels, -1, 3 * 3, 1, 1)


    def enforce_limits(self):
        # Enforce positive weights
        if self.mirror_weights:
            self.true_spatial_weight.data = F.softplus(self.true_spatial_weight, beta=10)
        else:
            self.spatial_weight.data = F.softplus(self.spatial_weight, beta=10)
        if self.in_channels > 1:
            self.channel_weight.data = F.softplus(self.channel_weight, beta=10)

    def forward(self, d, cd, e, ce):
        if self.mirror_weights:
            spatial_weight = torch.cat((self.true_spatial_weight, self.true_spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3).view(self.out_channels, -1, 9, 1, 1)
        else:
            spatial_weight = self.spatial_weight
            

        if self.in_channels > 1:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cd * d, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cd, self.channel_weight, groups=self.groups)
            d = nom / (denom + self.eps)
            cd = denom / (torch.sum(self.channel_weight) + self.eps)
        elif self.out_channels > 1:
            d, cd = d.expand(-1, self.out_channels,-1,-1), cd.expand(-1, self.out_channels,-1,-1)
            
        e_ext = F.pad(e, (1,1,1,1))
        e = e.clone()
        e_roll = torch.stack((e[:,:,0,::self.stride,::self.stride] * e_ext[:,:,0,:-2:self.stride,:-2:self.stride],  e[:,:,1,::self.stride,::self.stride] * e_ext[:,:,1,:-2:self.stride,1:-1:self.stride], e[:,:,2,::self.stride,::self.stride] * e_ext[:,:,2,:-2:self.stride,2::self.stride],
                              e[:,:,3,::self.stride,::self.stride] * e_ext[:,:,3,1:-1:self.stride,:-2:self.stride], torch.ones_like(e[:,:,1,::self.stride,::self.stride]),                                e[:,:,3,::self.stride,::self.stride] * e_ext[:,:,3,1:-1:self.stride,2::self.stride],
                              e[:,:,2,::self.stride,::self.stride] * e_ext[:,:,2,2::self.stride,:-2:self.stride],   e[:,:,1,::self.stride,::self.stride] * e_ext[:,:,1,2::self.stride,1:-1:self.stride],  e[:,:,0,::self.stride,::self.stride] * e_ext[:,:,0,2::self.stride,2::self.stride]), dim=2)
        d_roll = F.unfold(d,kernel_size=3,stride=self.stride, padding=1).view(e_roll.shape)  
        cd_roll = F.unfold(cd,kernel_size=3,stride=self.stride, padding=1).view(e_roll.shape)  

        # Normalized Convolution along spatial dimensions
        nom = F.conv3d(cd_roll * e_roll * d_roll, spatial_weight[:,0,None,:,:,:], groups=self.out_channels).squeeze(2)
        denom = F.conv3d(cd_roll * e_roll, spatial_weight[:,0,None,:,:,:], groups=self.out_channels).squeeze(2)
        d = nom / (denom + self.eps)
        cd = denom / (torch.sum(spatial_weight) + self.eps)  
        
        if self.kernel_size == 5:
            # second 3x3 conv
            # assume stride was 1, otherwise e_roll would have to be recalculated
            d_roll = F.unfold(d,kernel_size=3,stride=1, padding=1).view(e_roll.shape)  
            cd_roll = F.unfold(cd,kernel_size=3,stride=1, padding=1).view(e_roll.shape)  

            # Normalized Convolution along spatial dimensions
            nom = F.conv3d(cd_roll * e_roll * d_roll, spatial_weight[:,1,None,:,:,:], groups=self.out_channels).squeeze(2)
            denom = F.conv3d(cd_roll * e_roll, spatial_weight[:,1,None,:,:,:], groups=self.out_channels).squeeze(2)
            d = nom / (denom + self.eps)
            cd = denom / (torch.sum(spatial_weight) + self.eps)  

        return d, cd * self.devalue_conf
