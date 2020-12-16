
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
        if self.in_channels > self.out_channels:
            self.channel_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, 1, 1))     
        if mirror_weights:
            self.w_e_from_d = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 5, 1, 1))
            # / -> /  = \  -> \
            # / -> -  = \  -> -
            # / -> \  = \  -> /
            # / -> |  = \  -> |
            # - -> /  = -  -> \
            # - -> -  = sym
            # - -> \  = ^
            # - -> |  = sym
            # \ -> ...= ^
            # | -> /  = |  -> \
            # | -> -  = sym
            # | -> \  = ^
            # | -> |  = sym
            self.conv_weight_unsym = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 6, self.out_channels, self.kernel_size, self.kernel_size))
            self.conv_weight_sym = torch.nn.Parameter(data=torch.Tensor(self.out_channels,4,  self.out_channels, self.kernel_size, self.kernel_size // 2 + 1))
        else:
            self.w_e_from_d = torch.nn.Parameter(data=torch.Tensor(1, self.in_channels, 2, 4, 1, 1))
            self.conv_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels * 4, self.out_channels * 4, self.kernel_size, self.kernel_size))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.w_e_from_d) + 2
            torch.nn.init.xavier_uniform_(self.w_prop) + 1
            if self.in_channels > self.out_channels:
                torch.nn.init.xavier_uniform_(self.channel_weight) + 1
            if mirror_weights:
               torch.nn.init.xavier_uniform_(self.conv_weight_sym) + 1
               torch.nn.init.xavier_uniform_(self.conv_weight_unsym) + 1
            else:
               torch.nn.init.xavier_uniform_(self.conv_weight) + 1
        else:  # elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_e_from_d) + 1
            torch.nn.init.kaiming_uniform_(self.w_prop)
            if self.in_channels > self.out_channels:
                torch.nn.init.xavier_uniform_(self.channel_weight) + 1
            if mirror_weights:
               torch.nn.init.kaiming_uniform_(self.conv_weight_sym)
               torch.nn.init.kaiming_uniform_(self.conv_weight_unsym)
            else:
                torch.nn.init.kaiming_uniform_(self.conv_weight)
        if mirror_weights:
            self.conv_weight_sym.data[:,:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
            self.conv_weight_unsym.data[:,:,:, self.kernel_size // 2, self.kernel_size // 2] = 1
        else:
            self.conv_weight.data[:,:, self.kernel_size // 2, self.kernel_size // 2] = 1

            
    def enforce_limits(self):
        # Enforce positive weights
        if self.mirror_weights:
            self.conv_weight_sym.data = F.softplus(self.conv_weight_sym, beta=10)
            self.conv_weight_unsym.data = F.softplus(self.conv_weight_unsym, beta=10)
        else:
            self.conv_weight.data = F.softplus(self.conv_weight, beta=10)
        if self.in_channels > self.out_channels:
            self.channel_weight.data = F.softplus(self.channel_weight, beta=10)
        self.w_prop.data = F.softplus(self.w_prop, beta=10)
        self.w_e_from_d.data = F.softplus(self.w_e_from_d, beta=10)


    def forward(self, d, cd, e, ce):
        if self.mirror_weights:
            # / -> /  = \  -> \
            # / -> -  = \  -> -
            # / -> \  = \  -> /
            # / -> |  = \  -> |
            # - -> /  = -  -> \
            # - -> -  = sym
            # - -> \  = ^
            # - -> |  = sym
            # \ -> ...= ^
            # | -> /  = |  -> \
            # | -> -  = sym
            # | -> \  = ^
            # | -> |  = sym
            conv_weight_sym = torch.cat((self.conv_weight_sym, self.conv_weight_sym[:,:,:,:,:-1].flip(dims=(4,))), dim=4)

            conv_weight = torch.stack((self.conv_weight_unsym[:,:4,:,:,:].view(self.out_channels * 4, self.out_channels, self.kernel_size, self.kernel_size),
                                       torch.stack((self.conv_weight_unsym[:,4,:,:,:],
                                                   conv_weight_sym[:,0,:,:,:],
                                                   self.conv_weight_unsym[:,4,:,:,:].flip(dims=(3,)),
                                                   conv_weight_sym[:,1,:,:,:]), dim=1).view(self.out_channels * 4, self.out_channels, self.kernel_size, self.kernel_size),
                                       torch.stack((self.conv_weight_unsym[:,2,:,:,:].flip(dims=(3,)),
                                                   self.conv_weight_unsym[:,1,:,:,:].flip(dims=(3,)),
                                                   self.conv_weight_unsym[:,0,:,:,:].flip(dims=(3,)),
                                                   self.conv_weight_unsym[:,3,:,:,:].flip(dims=(3,))), dim=1).view(self.out_channels * 4, self.out_channels, self.kernel_size, self.kernel_size),
                                       torch.stack((self.conv_weight_unsym[:,5,:,:,:],
                                                   conv_weight_sym[:,2,:,:,:],
                                                   self.conv_weight_unsym[:,5,:,:,:].flip(dims=(3,)),
                                                   conv_weight_sym[:,3,:,:,:]), dim=1).view(self.out_channels * 4, self.out_channels, self.kernel_size, self.kernel_size) #
                                       ), dim=2).view(self.out_channels * 4, self.out_channels * 4, self.kernel_size, self.kernel_size)

            w_s_from_d = torch.stack((torch.stack((self.w_e_from_d[:,:,0], self.w_e_from_d[:,:,1]), dim=2),
                                      torch.stack((self.w_e_from_d[:,:,2], self.w_e_from_d[:,:,3]), dim=2),
                                      torch.stack((self.w_e_from_d[:,:,1], self.w_e_from_d[:,:,0]), dim=2),
                                      torch.stack((self.w_e_from_d[:,:,4], self.w_e_from_d[:,:,4]), dim=2)), dim=3)
        else:
            conv_weight = self.conv_weight
            w_s_from_d = self.w_e_from_d



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
        e = ((self.w_prop * ce * e + 1 * ce_from_d * e_from_d) / (self.w_prop * ce + 1 * ce_from_d + self.eps))
        ce = ((self.w_prop * ce + 1 * ce_from_d) / (self.w_prop + 1))

        if self.in_channels > self.out_channels:
            # channel-wise convolution
            e, ce = e.view(real_shape[0],self.in_channels,4 * real_shape[2], real_shape[3]), ce.view(real_shape[0],self.in_channels,4 * real_shape[2], real_shape[3])
            nom = F.conv2d(ce * e, self.channel_weight)
            denom = F.conv2d(ce, self.channel_weight)
            e = nom / (denom + self.eps)
            ce = denom / (torch.sum(self.channel_weight) + self.eps)


        # Normalized Convolution
        e, ce = e.view(real_shape[0], self.out_channels* 4, real_shape[2], real_shape[3]), ce.view(real_shape[0], self.out_channels* 4, real_shape[2], real_shape[3])
        nom = F.conv2d(ce * e, conv_weight, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        denom = F.conv2d(ce, conv_weight, stride=self.stride, padding=self.padding, dilation=self.dilation).view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
        e = nom / (denom + self.eps)
        ce = denom / (torch.sum(conv_weight) + self.eps)

        return e.view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3]), ce.view(real_shape[0], self.out_channels, 4, real_shape[2], real_shape[3])
