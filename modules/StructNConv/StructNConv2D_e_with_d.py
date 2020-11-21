
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

class StructNConv2D_e_with_d(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, 
                 in_channels=1, out_channels=1, groups=1, kernel_size=1, stride=1, padding=0, dilation=1, devalue_pooled_confidence=True):
        super(StructNConv2D_e_with_d, self).__init__()

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
        self.w_s_from_d = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, 1))
        self.w_prop = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 1, 1, 1))
        self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels*4, self.in_channels*4, 1, 1))
        if mirror_weights:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels*4, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels*4, 1, self.kernel_size, self.kernel_size))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_from_d)
            torch.nn.init.xavier_uniform_(self.w_prop) + 1
            torch.nn.init.xavier_uniform_(self.channel_weight) + 1
            torch.nn.init.xavier_uniform_(spatial_weight) + 1
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_from_d)
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(spatial_weight)
        spatial_weight.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
        
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
        self.channel_weight.data = F.softplus(self.channel_weight, beta=10)
        self.w_prop.data = F.softplus(self.w_prop, beta=10)
        self.w_s_from_d.data = torch.sigmoid(self.w_s_from_d)


    def forward(self, d, cd, s, cs):
        if self.mirror_weights:
            self.spatial_weight = torch.cat((self.true_spatial_weight, self.true_spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)

        # calculate smoothness from depths
        #edges:
        #0 => /
        #1 => -
        #2 => \
        #3 => |
        # dim 2 in 0 to 4
        real_shape=d.shape
        d, cd = F.pad(d, (1,1,1,1)), F.pad(cd, (1,1,1,1))
        shape = d.shape

        if self.kernel_size == 3:
            d_min=d_max=d
            cd_min=cd_max=cd
        else:
            _, j_max = F.max_pool2d(d * cd, kernel_size=3, stride=1, return_indices=True, padding=1)
            _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=3, stride=1, return_indices=True, padding=1)
            d_min, d_max = retrieve_indices(d, j_min), retrieve_indices(d, j_max)
            cd_min, cd_max = retrieve_indices(cd, j_min), retrieve_indices(cd, j_max)
        
        # d_min on one side divided by d_max on the other = low value => edge in the middle
        # clamp is needed to prevent min > max, which can happen because they originate in different regions
        # ordinarely, this would still maintain min <= overlap <= max, but because of ^, d_min might chooes an initialized value over the true min of 0
        d_min_div_max = torch.clamp(torch.stack((torch.stack((d_min[:,:,:-2,:-2] / (d_max[:,:,2:,2:] + self.eps),d_min[:,:,:-2,:-2] / (d_max[:,:,2:,2:] + self.eps)), dim=2),        
                                    torch.stack((d_min[:,:,:-2,1:-1] / (d_max[:,:,2:,1:-1] + self.eps),d_min[:,:,:-2,1:-1] / (d_max[:,:,2:,1:-1] + self.eps)), dim=2),
                                    torch.stack((d_min[:,:,:-2,2:] / (d_max[:,:,2:,:-2] + self.eps),d_min[:,:,:-2,2:] / (d_max[:,:,2:,:-2] + self.eps)), dim=2),
                                    torch.stack((d_min[:,:,1:-1,2:] / (d_max[:,:,1:-1,:-2] + self.eps),d_min[:,:,1:-1,2:] / (d_max[:,:,1:-1,:-2] + self.eps)), dim=2)),dim=2),0,1)

        c_min_div_max = torch.stack((torch.stack((cd_min[:,:,:-2,:-2] * cd_max[:,:,2:,2:], cd_min[:,:,:-2,:-2] * cd_max[:,:,2:,2:]), dim=2),
                                     torch.stack((cd_min[:,:,:-2,1:-1] * cd_max[:,:,2:,1:-1], cd_min[:,:,:-2,1:-1] * cd_max[:,:,2:,1:-1]), dim=2),
                                     torch.stack((cd_min[:,:,:-2,2:] * cd_max[:,:,2:,:-2], cd_min[:,:,:-2,2:] * cd_max[:,:,2:,:-2]), dim=2),
                                     torch.stack((cd_min[:,:,1:-1,2:] * cd_max[:,:,1:-1,:-2], cd_min[:,:,1:-1,2:] * cd_max[:,:,1:-1,:-2]), dim=2)), dim=2)
        
        j_min = torch.argmax(c_min_div_max / (d_min_div_max + self.eps), dim=3, keepdim=True)
        
        min_div_max, cs_from_d = d_min_div_max.gather(index=j_min, dim=2).squeeze(3), c_min_div_max.gather(index=j_min, dim=2).squeeze(3)
        s_from_d = (1 - self.w_s_from_d) * min_div_max + self.w_s_from_d * min_div_max ** 2

        # combine with previous smoothness
        s_prop = ((self.w_prop * cs * s + 1 * cs_from_d * s_from_d) / (self.w_prop * cs + 1 * cs_from_d + self.eps)).view(real_shape[0],-1,real_shape[2], real_shape[3])
        cs_prop = ((self.w_prop * cs + 1 * cs_from_d) / (self.w_prop + 1)).view(real_shape[0],-1,real_shape[2], real_shape[3])

        
        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(cs_prop * s_prop, self.spatial_weight, groups=self.in_channels*4, stride=self.stride, padding=self.padding, dilation=self.dilation).squeeze(2)
        denom = F.conv2d(cs_prop, self.spatial_weight, groups=self.in_channels*4, stride=self.stride, padding=self.padding, dilation=self.dilation).squeeze(2)
        s_spatial = nom / (denom + self.eps)
        cs_spatial = denom / (torch.sum(self.spatial_weight) + self.eps)

        # Normalized Convolution along channel dimensions
        nom = F.conv2d(cs_spatial * s_spatial, self.channel_weight, groups=self.groups)
        denom = F.conv2d(cs_spatial, self.channel_weight, groups=self.groups)
        s = (nom / (denom + self.eps)).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        cs = (denom / (torch.sum(self.channel_weight) + self.eps)).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])

        return s, cs