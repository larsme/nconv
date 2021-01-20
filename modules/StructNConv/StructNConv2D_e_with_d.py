
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
        self.w_prop = torch.nn.Parameter(data=torch.ones(1, self.in_channels, 1, 1, 1))
        if mirror_weights:
            self.w_pow = torch.nn.Parameter(data=torch.ones(1, self.in_channels, 5, 1, 1))
            self.spatial_weight0 = torch.nn.Parameter(data=torch.ones(self.out_channels, 1, self.kernel_size, self.kernel_size))
            self.spatial_weight1 = torch.nn.Parameter(data=torch.ones(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
            self.spatial_weight3 = torch.nn.Parameter(data=torch.ones(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
            self.channel_weight = torch.nn.Parameter(data=torch.ones(self.out_channels, self.in_channels, 10))
        else:
            self.w_pow = torch.nn.Parameter(data=torch.ones(1, self.in_channels, 2, 4, 1, 1))
            self.spatial_weight = torch.nn.Parameter(data=torch.ones(self.out_channels * 4, 1, self.kernel_size, self.kernel_size))
            self.channel_weight = torch.nn.Parameter(data=torch.ones(self.out_channels * 4, self.in_channels * 4, 1, 1))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_pow)
            torch.nn.init.xavier_uniform_(self.w_prop) 
            torch.nn.init.xavier_uniform_(self.channel_weight) 
            if mirror_weights:
               torch.nn.init.xavier_uniform_(self.spatial_weight0) 
               torch.nn.init.xavier_uniform_(self.spatial_weight1) 
               torch.nn.init.xavier_uniform_(self.spatial_weight3) 
            else:
               torch.nn.init.xavier_uniform_(self.spatial_weight) 
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_pow) 
            torch.nn.init.kaiming_uniform_(self.w_prop)
            torch.nn.init.kaiming_uniform_(self.channel_weight)
            if mirror_weights:
               torch.nn.init.kaiming_uniform_(self.spatial_weight0)
               torch.nn.init.kaiming_uniform_(self.spatial_weight1)
               torch.nn.init.kaiming_uniform_(self.spatial_weight3)
            else:
                torch.nn.init.kaiming_uniform_(self.spatial_weight)
        
    def print(self, s_list):
        spatial_weight, channel_weight, w_pow, w_prop = self.prepare_weights()

        for dir in range(4):
            for x in range(self.kernel_size):
                for y in range(self.kernel_size):
                    s_list[y + dir * 6+1] += '{0:0.2f}, '.format(spatial_weight[dir,0,y,x].item())

        idx = 4 * 6 + 2
        for channel in range(self.in_channels):
            for dir in range(4):
                for d2 in range(4):
                    s_list[dir + idx + channel * 5] += '{0:0.2f}, '.format(channel_weight[dir,channel * 4 + d2,0, 0].item())

        idx += 11
        for channel in range(self.in_channels):
            for dir in range(4):
                for side in range(2):
                    s_list[idx + dir + channel * 5] += '{0:0.2f}, '.format(w_pow[0,channel,side,dir].item())

        idx += 11
        for channel in range(self.in_channels):
            s_list[idx + channel] += '{0:0.2f}, '.format(w_prop[0, channel, 0,0].item())

        return s_list

    def prepare_weights(self):
        if self.mirror_weights:
            spatial_weight0 = F.softplus(self.spatial_weight0)
            spatial_weight1 = F.softplus(self.spatial_weight1)
            spatial_weight3 = F.softplus(self.spatial_weight3)
            spatial_weight = torch.cat((spatial_weight0,
                                        torch.cat((spatial_weight1, spatial_weight1[:,:,:,:-1].flip(dims=(3,))), dim=3),
                                        spatial_weight0.flip(dims=(3,)),
                                        torch.cat((spatial_weight3, spatial_weight3[:,:,:,:-1].flip(dims=(3,))), dim=3)), dim=0)
            # / -> |  = \  -> |
            # / -> -  = \  -> -
            # | -> /  = |  -> \
            # - -> /  = -  -> \
            # / -> /  = \  -> \
            # / -> \  = \  -> /
            channel_weight = F.softplus(self.channel_weight)
            channel_weight = torch.stack((torch.stack((channel_weight[:,:,0], channel_weight[:,:,1], channel_weight[:,:,2], channel_weight[:,:,3]), dim = 1),
                                          torch.stack((channel_weight[:,:,4], channel_weight[:,:,5], channel_weight[:,:,4], channel_weight[:,:,6]), dim = 1),
                                          torch.stack((channel_weight[:,:,2], channel_weight[:,:,1], channel_weight[:,:,0], channel_weight[:,:,3]), dim = 1),
                                          torch.stack((channel_weight[:,:,7], channel_weight[:,:,8], channel_weight[:,:,7], channel_weight[:,:,9]), dim = 1)), 
                                         dim=3).view(self.out_channels * 4, self.in_channels * 4, 1, 1)
            w_pow = F.softplus(self.w_pow)
            w_pow = torch.stack((torch.stack((w_pow[:,:,0], w_pow[:,:,1]), dim=2),
                                 torch.stack((w_pow[:,:,2], w_pow[:,:,3]), dim=2),
                                 torch.stack((w_pow[:,:,1], w_pow[:,:,0]), dim=2),
                                 torch.stack((w_pow[:,:,4], w_pow[:,:,4]), dim=2)), dim=3)
        else:
            spatial_weight = F.softplus(self.spatial_weight)
            channel_weight = F.softplus(self.channel_weight)
            w_pow = F.softplus(self.w_pow)
        spatial_weight = spatial_weight / spatial_weight.sum(dim=[2,3], keepdim=True)
        channel_weight = channel_weight / channel_weight.sum(dim=1, keepdim=True)
        w_prop = torch.sigmoid(self.w_prop)

        return spatial_weight, channel_weight, w_pow, w_prop

    
    def prep_eval(self):
        self.weights = self.prepare_weights()
        
    def forward(self, d, cd, e, ce):
        if self.training:
            spatial_weight, channel_weight, w_pow, w_prop = self.prepare_weights()
        else:
            spatial_weight, channel_weight, w_pow, w_prop = self.weights
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
            # d on one side divided by d on the other = low value => edge in the middle
            e_from_d = torch.min(torch.stack(((d[:,:, :-2, :-2] + self.eps) / (d[:,:,2:  ,2:] + self.eps),        
                                              (d[:,:, :-2,1:-1] + self.eps) / (d[:,:,2:  ,1:-1] + self.eps),
                                              (d[:,:, :-2,2:] + self.eps) / (d[:,:,2:  , :-2] + self.eps),
                                              (d[:,:,1:-1,2:] + self.eps) / (d[:,:,1:-1, :-2] + self.eps)), dim=2).clamp_max(1).pow(w_pow[:,:,0,:,:,:]),
                                 torch.stack(((d[:,:,2:  ,2:] + self.eps) / (d[:,:, :-2, :-2] + self.eps),        
                                              (d[:,:,2:  ,1:-1] + self.eps) / (d[:,:, :-2,1:-1] + self.eps),
                                              (d[:,:,2:  , :-2] + self.eps) / (d[:,:, :-2,2:] + self.eps),
                                              (d[:,:,1:-1, :-2] + self.eps) / (d[:,:,1:-1,2:] + self.eps)), dim=2).clamp_max(1).pow(w_pow[:,:,1,:,:,:]))

            ce_from_d = torch.stack((cd[:,:, :-2, :-2] * cd[:,:,2:  ,2:],
                                     cd[:,:, :-2,1:-1] * cd[:,:,2:  ,1:-1],
                                     cd[:,:, :-2,2:] * cd[:,:,2:  , :-2],
                                     cd[:,:,1:-1,2:] * cd[:,:,1:-1, :-2]), dim=2)
        else:
            _, j_max = F.max_pool2d(d * cd, kernel_size=3, stride=1, return_indices=True, padding=1)
            _, j_min = F.max_pool2d(cd / (d + self.eps), kernel_size=3, stride=1, return_indices=True, padding=1)
            d_min, d_max = retrieve_indices(d, j_min), retrieve_indices(d, j_max)
            cd_min, cd_max = retrieve_indices(cd, j_min), retrieve_indices(cd, j_max)
        
            # d_min on one side divided by d_max on the other = low value => edge in the middle
            # clamp to 1 is needed to prevent min > max, which can happen with confidence weighting because they originate in different regions
            d_min_div_max = torch.clamp(torch.stack((torch.stack((d_min[:,:, :-2, :-2] / (d_max[:,:,2:  ,2:] + self.eps), d_min[:,:,2:  ,2:] / (d_max[:,:, :-2, :-2] + self.eps)), dim=2),        
                                                     torch.stack((d_min[:,:, :-2,1:-1] / (d_max[:,:,2:  ,1:-1] + self.eps), d_min[:,:,2:  ,1:-1] / (d_max[:,:, :-2,1:-1] + self.eps)), dim=2),
                                                     torch.stack((d_min[:,:, :-2,2:] / (d_max[:,:,2:  , :-2] + self.eps), d_min[:,:,2:  , :-2] / (d_max[:,:, :-2,2:] + self.eps)), dim=2),
                                                     torch.stack((d_min[:,:,1:-1,2:] / (d_max[:,:,1:-1, :-2] + self.eps), d_min[:,:,1:-1, :-2] / (d_max[:,:,1:-1,2:] + self.eps)), dim=2)),
                                                    dim=3),self.eps,1).pow(w_pow)

            c_min_div_max = torch.stack((torch.stack((cd_min[:,:, :-2, :-2] * cd_max[:,:,2:  ,2:], cd_max[:,:, :-2, :-2] * cd_min[:,:,2:  ,2:]), dim=2),
                                         torch.stack((cd_min[:,:, :-2,1:-1] * cd_max[:,:,2:  ,1:-1], cd_max[:,:, :-2,1:-1] * cd_min[:,:,2:  ,1:-1]), dim=2),
                                         torch.stack((cd_min[:,:, :-2,2:] * cd_max[:,:,2:  , :-2], cd_max[:,:, :-2,2:] * cd_min[:,:,2:  , :-2]), dim=2),
                                         torch.stack((cd_min[:,:,1:-1,2:] * cd_max[:,:,1:-1, :-2], cd_max[:,:,1:-1,2:] * cd_min[:,:,1:-1, :-2]), dim=2)), dim=3)
        
            # dim 2 contains d_min(side1) / d_max(side2) and d_min(side2) / d_max(side1)
            # take the confidence weighted min of both and gather the respective items
            j_min = torch.argmax(c_min_div_max / d_min_div_max, dim=2, keepdim=True)
            e_from_d = d_min_div_max.gather(index=j_min, dim=2).squeeze(2)
            ce_from_d = c_min_div_max.gather(index=j_min, dim=2).squeeze(2)

        # combine with previous smoothness
        e = ((w_prop * ce * e + (1 - w_prop) * ce_from_d * e_from_d) / (w_prop * ce + (1 - w_prop) * ce_from_d + self.eps)).view(real_shape[0],-1,real_shape[2], real_shape[3])
        ce = (w_prop * ce + (1 - w_prop) * ce_from_d).view(real_shape[0],-1,real_shape[2], real_shape[3])

        # Normalized Convolution along channel dimensions
        nom = F.conv2d(ce * e, channel_weight, groups=self.groups)
        ce = F.conv2d(ce, channel_weight, groups=self.groups)
        e = (nom / (ce + self.eps))

        
        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(ce * e, spatial_weight, groups=self.out_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        ce = F.conv2d(ce, spatial_weight, groups=self.out_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        e = nom / (ce + self.eps)

        return e, ce
