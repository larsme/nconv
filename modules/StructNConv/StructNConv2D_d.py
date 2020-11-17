
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch
import torch.nn.functional as F


class StructNConv2D_d(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, in_channels=1, out_channels=1, groups=1,
                 kernel_size=1, stride=1, padding=0, dilation=1, devalue_pooled_confidence=True):
        super(StructNConv2D_d, self).__init__()

        self.eps = 1e-20
        self.init_method = init_method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.mirror_weights = mirror_weights

        self.devalue_conf = 1 / self.stride / self.stride if devalue_pooled_confidence else 1

        # Define Parameters
        if self.in_channels > 1:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 1, 1))
        if mirror_weights:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, self.kernel_size))

        # Init Parameters
        if 'x' in self.init_method:  # Xavier
            if self.in_channels > 1:
                torch.nn.init.xavier_uniform_(self.channel_weight) + 1
            torch.nn.init.xavier_uniform_(spatial_weight) + 1
        elif 'k' in self.init_method: # Kaiming
            if self.in_channels > 1:
                torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(spatial_weight)
        if 'n' in self.init_method:
            spatial_weight.data[-1,:, self.kernel_size // 2, self.kernel_size // 2] = 3
            if in_channels > 1:
                spatial_weight.data[-1,:,:,:] = 1
        
        if mirror_weights:
            self.true_spatial_weight = spatial_weight
        else:
            self.spatial_weight = spatial_weight
            
    def enforce_limits(self):
        # Enforce positive weights
        if self.mirror_weights:
            self.true_spatial_weight.data = F.softplus(self.true_spatial_weight, beta=10)
        else:
            self.spatial_weight.data = F.softplus(self.spatial_weight, beta=10)
        if self.in_channels > 1:
            self.channel_weight.data = F.softplus(self.channel_weight, beta=10)

    def forward(self, d, cd):
        if self.mirror_weights:
            self.spatial_weight = torch.cat((self.true_spatial_weight, self.true_spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)

        # Normalized Convolution along channel dimensions
        if self.in_channels > 1:
            nom = F.conv2d(cd * d, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cd, self.channel_weight, groups=self.groups)
            d = nom / (denom + self.eps)
            cd = denom / (torch.sum(self.channel_weight) + self.eps)
        elif self.out_channels > 1:
            d, cd = d.expand(-1, self.out_channels,-1,-1), cd.expand(-1, self.out_channels,-1,-1)

        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(cd * d, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)
        denom = F.conv2d(cd, self.spatial_weight, groups=self.out_channels, stride=self.stride,
                            padding=self.padding, dilation=self.dilation)
        d = nom / (denom + self.eps)
        cd = denom / (torch.sum(self.spatial_weight) + self.eps)

        if self.devalue_conf!=1:
            return d, cd * self.devalue_conf
        else:
            return d, cd
