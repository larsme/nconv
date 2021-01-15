
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

class StructNConv2D_s_with_d(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, 
                 in_channels=1, out_channels=1, groups=1, kernel_size=1, stride=1, padding=0, dilation=1, devalue_pooled_confidence=True):
        super(StructNConv2D_s_with_d, self).__init__()

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
        self.w_s_from_d = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 1, 1))
        self.w_prop = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 1, 1))
        if self.in_channels > 1:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 1, 1))
        if mirror_weights:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size, self.kernel_size))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if self.in_channels > 1:
                torch.nn.init.xavier_uniform_(self.channel_weight)
            torch.nn.init.xavier_uniform_(spatial_weight)
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_from_d)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            if self.in_channels > 1:
                torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(spatial_weight)
        
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
        self.w_prop.data = F.softplus(self.w_prop, beta=10)
        self.w_s_from_d.data = torch.clamp(self.w_s_from_d, -1, 1)


    def forward(self, d, cd, s, cs):
        if self.mirror_weights:
            spatial_weight = torch.cat((self.true_spatial_weight, self.true_spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)
        else:
            spatial_weight = self.spatial_weight

        # calculate smoothness from depths
        _, j_max = F.max_pool2d(d * cd, kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)
        _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)

        min_div_max = retrieve_indices(d, j_min) / (retrieve_indices(d, j_max) + self.eps)

        s_from_d = (1 - self.w_s_from_d) * min_div_max + self.w_s_from_d * min_div_max ** 2
        cs_from_d = retrieve_indices(cd, j_max) * retrieve_indices(cd, j_min)

        if self.stride == 1:
            s_prop = (self.w_prop * cs * s + 1 * cs_from_d * s_from_d) / (self.w_prop * cs + 1 * cs_from_d + self.eps)
            cs_prop = (self.w_prop * cs + 1 * cs_from_d) / (self.w_prop + 1)
        else:
            s_prop = s
            cs_prop = cs

        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(cs_prop * s_prop, spatial_weight, groups=self.in_channels, stride=self.stride,
                        padding=self.padding, dilation=self.dilation).squeeze(2)
        denom = F.conv2d(cs_prop, spatial_weight, groups=self.in_channels, stride=self.stride,
                            padding=self.padding, dilation=self.dilation).squeeze(2)
        s = nom / (denom + self.eps)
        cs = denom / (torch.sum(spatial_weight) + self.eps)
        
        if self.in_channels > 1:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cs * s, self.channel_weight, groups=self.groups)
            denom = F.conv2d(cs, self.channel_weight, groups=self.groups)
            s = nom / (denom + self.eps)
            cs = denom / (torch.sum(self.channel_weight) + self.eps)
        elif self.out_channels > 1:
            s, cs = s.expand(-1, self.out_channels,-1,-1), cs.expand(-1, self.out_channels,-1,-1)

        if self.stride > 1:
            s = (self.w_prop * cs * s + 1 * cs_from_d * s_from_d) / (self.w_prop * cs + 1 * cs_from_d + self.eps)
            cs = (self.w_prop * cs + 1 * cs_from_d) / (self.w_prop + 1)
            
            return s, cs * self.devalue_conf
        else:
            return s, cs
