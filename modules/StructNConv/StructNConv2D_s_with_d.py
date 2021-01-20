
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
        self.w_pow = torch.nn.Parameter(data=torch.ones(1, self.in_channels, 1, 1))
        self.w_prop = torch.nn.Parameter(data=torch.ones(1, self.in_channels, 1, 1))
        if self.in_channels > 1:
            self.channel_weight = torch.nn.Parameter(data=torch.ones(self.out_channels, self.in_channels, 1, 1))
        if mirror_weights:
            self.spatial_weight = torch.nn.Parameter(data=torch.ones(self.in_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            self.spatial_weight = torch.nn.Parameter(data=torch.ones(self.in_channels, 1, self.kernel_size, self.kernel_size))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_pow)
            torch.nn.init.xavier_uniform_(self.w_prop)
            if self.in_channels > 1:
                torch.nn.init.xavier_uniform_(self.channel_weight) 
            torch.nn.init.xavier_uniform_(self.spatial_weight) 
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_pow)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            if self.in_channels > 1:
                torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(self.spatial_weight)

    def prepare_weights(self):
        if self.mirror_weights:
            spatial_weight = F.softplus(self.spatial_weight)
            spatial_weight = torch.cat((spatial_weight, spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)
        else:
            spatial_weight = F.softplus(self.spatial_weight)
        spatial_weight = spatial_weight / spatial_weight.sum(dim=[2,3], keepdim=True)
        w_prop = torch.sigmoid(self.w_prop)
        w_pow = F.softplus(self.w_pow)

        if self.in_channels > 1:
            channel_weight = F.softplus(self.channel_weight)
            channel_weight = channel_weight / channel_weight.sum(dim=1, keepdim=True)
        else:
            channel_weight = None

        return spatial_weight, channel_weight, w_prop, w_pow
    
    def prep_eval(self):
        self.weights = self.prepare_weights()
        
    def forward(self, d, cd, s, cs):
        if self.training:
            spatial_weight, channel_weight, w_prop, w_pow = self.prepare_weights()
        else:
            spatial_weight, channel_weight, w_prop, w_pow = self.weights

        # calculate smoothness from depths
        _, j_max = F.max_pool2d(d * cd, kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)
        _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=self.kernel_size, stride=self.stride,
                                return_indices=True, padding=self.padding)

        min_div_max = retrieve_indices(d, j_min) / (retrieve_indices(d, j_max) + self.eps)

        s_from_d = min_div_max.pow(w_pow)
        cs_from_d = retrieve_indices(cd, j_max) * retrieve_indices(cd, j_min)

        if self.stride == 1:
            s_prop = (w_prop * cs * s + (1 - w_prop) * cs_from_d * s_from_d) / (w_prop * cs + (1 - w_prop) * cs_from_d + self.eps)
            cs_prop = (w_prop * cs + (1 - w_prop) * cs_from_d)
        else:
            s_prop = s
            cs_prop = cs

        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(cs_prop * s_prop, spatial_weight, groups=self.in_channels, stride=self.stride,
                        padding=self.padding, dilation=self.dilation).squeeze(2)
        cs = F.conv2d(cs_prop, spatial_weight, groups=self.in_channels, stride=self.stride,
                            padding=self.padding, dilation=self.dilation).squeeze(2)
        s = nom / (cs + self.eps)
        
        if self.in_channels > 1:
            # Normalized Convolution along channel dimensions
            nom = F.conv2d(cs * s, channel_weight, groups=self.groups)
            cs = F.conv2d(cs, channel_weight, groups=self.groups)
            s = nom / (cs + self.eps)
        elif self.out_channels > 1:
            s, cs = s.expand(-1, self.out_channels,-1,-1), cs.expand(-1, self.out_channels,-1,-1)

        if self.stride > 1:
            s = (w_prop * cs * s + (1 - w_prop) * cs_from_d * s_from_d) / (w_prop * cs + (1 - w_prop) * cs_from_d + self.eps)
            cs = (w_prop * cs + (1 - w_prop) * cs_from_d)
            
            return s, cs * self.devalue_conf
        else:
            return s, cs
