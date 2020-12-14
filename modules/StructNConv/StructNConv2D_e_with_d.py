
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
        self.w_prop = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 1, 1, 1))
        if mirror_weights:
            self.w_s_from_d = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 5, 1, 1))
            self.spatial_weight0 = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, self.kernel_size))
            self.spatial_weight1 = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
            self.spatial_weight3 = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 10))
        else:
            self.w_s_from_d = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 2, 4, 1, 1))
            self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels * 4, 1, self.kernel_size, self.kernel_size))
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels * 4, self.in_channels * 4, 1, 1))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_s_from_d) + 2
            torch.nn.init.xavier_uniform_(self.w_prop) + 1
            torch.nn.init.xavier_uniform_(self.channel_weight) + 1
            if mirror_weights:
               torch.nn.init.xavier_uniform_(self.spatial_weight0) + 1
               torch.nn.init.xavier_uniform_(self.spatial_weight1) + 1
               torch.nn.init.xavier_uniform_(self.spatial_weight3) + 1
            else:
               torch.nn.init.xavier_uniform_(self.spatial_weight) + 1
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_s_from_d) + 1
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            if mirror_weights:
               torch.nn.init.kaiming_uniform_(self.spatial_weight0)
               torch.nn.init.kaiming_uniform_(self.spatial_weight1)
               torch.nn.init.kaiming_uniform_(self.spatial_weight3)
            else:
                torch.nn.init.kaiming_uniform_(self.spatial_weight)
        if mirror_weights:
            self.spatial_weight0.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
            self.spatial_weight1.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
            self.spatial_weight3.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
        else:
            self.spatial_weight.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1

            
    def enforce_limits(self):
        # Enforce positive weights
        if self.mirror_weights:
            self.spatial_weight0.data = F.softplus(self.spatial_weight0, beta=10)
            self.spatial_weight1.data = F.softplus(self.spatial_weight1, beta=10)
            self.spatial_weight3.data = F.softplus(self.spatial_weight3, beta=10)
        else:
            self.spatial_weight.data = F.softplus(self.spatial_weight, beta=10)
        self.channel_weight.data = F.softplus(self.channel_weight, beta=10)
        self.w_prop.data = F.softplus(self.w_prop, beta=10)
        self.w_s_from_d.data = F.softplus(self.w_s_from_d, beta=10)


    def forward(self, d, cd, e, ce):
        if self.mirror_weights:
            spatial_weight = torch.cat((self.spatial_weight0,
                                        torch.cat((self.spatial_weight1, self.spatial_weight1[:,:,:,:-1].flip(dims=(3,))), dim=3),
                                        self.spatial_weight0.flip(dims=(3,)),
                                        torch.cat((self.spatial_weight3, self.spatial_weight3[:,:,:,:-1].flip(dims=(3,))), dim=3)), dim=0)
            # / -> |  = \  -> |
            # / -> -  = \  -> -
            # | -> /  = |  -> \
            # - -> /  = -  -> \
            # / -> /  = \  -> \
            # / -> \  = \  -> /
            channel_weight = torch.stack((torch.stack((self.channel_weight[:,:,0], self.channel_weight[:,:,1], self.channel_weight[:,:,2], self.channel_weight[:,:,3]), dim = 1),
                                          torch.stack((self.channel_weight[:,:,4], self.channel_weight[:,:,5], self.channel_weight[:,:,4], self.channel_weight[:,:,6]), dim = 1),
                                          torch.stack((self.channel_weight[:,:,2], self.channel_weight[:,:,1], self.channel_weight[:,:,0], self.channel_weight[:,:,3]), dim = 1),
                                          torch.stack((self.channel_weight[:,:,7], self.channel_weight[:,:,8], self.channel_weight[:,:,7], self.channel_weight[:,:,9]), dim = 1)), 
                                         dim=3).view(self.out_channels * 4, self.in_channels * 4, 1, 1)
            w_s_from_d = torch.stack((torch.stack((self.w_s_from_d[:,:,0], self.w_s_from_d[:,:,1]), dim=2),
                                      torch.stack((self.w_s_from_d[:,:,2], self.w_s_from_d[:,:,3]), dim=2),
                                      torch.stack((self.w_s_from_d[:,:,1], self.w_s_from_d[:,:,0]), dim=2),
                                      torch.stack((self.w_s_from_d[:,:,4], self.w_s_from_d[:,:,4]), dim=2)), dim=3)
        else:
            spatial_weight = self.spatial_weight
            channel_weight = self.channel_weight
            w_s_from_d = self.w_s_from_d



        # calculate smoothness from depths
        #edges:
        #0 => /
        #1 => -
        #2 => \
        #3 => |
        # dim 2 in 0 to 4
        real_shape = d.shape
        d, cd = F.pad(d, (1,1,1,1)), F.pad(cd, (1,1,1,1))
        shape = d.shape

        if self.kernel_size == 3:
            d_min = d_max = d
            cd_min = cd_max = cd
        else:
            _, j_max = F.max_pool2d(d * cd, kernel_size=3, stride=1, return_indices=True, padding=1)
            _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=3, stride=1, return_indices=True, padding=1)
            d_min, d_max = retrieve_indices(d, j_min), retrieve_indices(d, j_max)
            cd_min, cd_max = retrieve_indices(cd, j_min), retrieve_indices(cd, j_max)
        
        # d_min on one side divided by d_max on the other = low value => edge in the middle
        # clamp is needed to prevent min > max, which can happen because they originate in different regions
        # ordinarely, this would still maintain min <= overlap <= max, but because of ^, d_min might chooes an initialized value over the true min of 0
        d_min_div_max = torch.clamp(torch.stack((torch.stack((d_min[:,:, :-2, :-2] / (d_max[:,:,2:  ,2:] + self.eps), d_max[:,:, :-2, :-2] / (d_min[:,:,2:  ,2:] + self.eps)), dim=2),        
                                                 torch.stack((d_min[:,:, :-2,1:-1] / (d_max[:,:,2:  ,1:-1] + self.eps), d_max[:,:, :-2,1:-1] / (d_min[:,:,2:  ,1:-1] + self.eps)), dim=2),
                                                 torch.stack((d_min[:,:, :-2,2:] / (d_max[:,:,2:  , :-2] + self.eps), d_max[:,:, :-2,2:] / (d_min[:,:,2:  , :-2] + self.eps)), dim=2),
                                                 torch.stack((d_min[:,:,1:-1,2:] / (d_max[:,:,1:-1, :-2] + self.eps), d_max[:,:,1:-1,2:] / (d_min[:,:,1:-1, :-2] + self.eps)), dim=2)),
                                                dim=3),self.eps,1).pow(w_s_from_d)

        c_min_div_max = torch.stack((torch.stack((cd_min[:,:, :-2, :-2] * cd_max[:,:,2:  ,2:], cd_max[:,:, :-2, :-2] * cd_min[:,:,2:  ,2:]), dim=2),
                                     torch.stack((cd_min[:,:, :-2,1:-1] * cd_max[:,:,2:  ,1:-1], cd_max[:,:, :-2,1:-1] * cd_min[:,:,2:  ,1:-1]), dim=2),
                                     torch.stack((cd_min[:,:, :-2,2:] * cd_max[:,:,2:  , :-2], cd_max[:,:, :-2,2:] * cd_min[:,:,2:  , :-2]), dim=2),
                                     torch.stack((cd_min[:,:,1:-1,2:] * cd_max[:,:,1:-1, :-2], cd_max[:,:,1:-1,2:] * cd_min[:,:,1:-1, :-2]), dim=2)), dim=3)
        
        j_min = torch.argmax(c_min_div_max / d_min_div_max, dim=2, keepdim=True)

        e_from_d = d_min_div_max.gather(index=j_min, dim=2).squeeze(2)
        ce_from_d = c_min_div_max.gather(index=j_min, dim=2).squeeze(2)

        # combine with previous smoothness
        e = ((self.w_prop * ce * e + 1 * ce_from_d * e_from_d) / (self.w_prop * ce + 1 * ce_from_d + self.eps)).view(real_shape[0],-1,real_shape[2], real_shape[3])
        ce = ((self.w_prop * ce + 1 * ce_from_d) / (self.w_prop + 1)).view(real_shape[0],-1,real_shape[2], real_shape[3])

        # Normalized Convolution along channel dimensions
        nom = F.conv2d(ce * e, channel_weight, groups=self.groups)
        denom = F.conv2d(ce, channel_weight, groups=self.groups)
        e = (nom / (denom + self.eps))
        ce = (denom / (torch.sum(channel_weight) + self.eps))

        
        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(ce * e, spatial_weight, groups=self.out_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        denom = F.conv2d(ce, spatial_weight, groups=self.out_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        e = nom / (denom + self.eps)
        ce = denom / (torch.sum(spatial_weight) + self.eps)

        return e, ce
