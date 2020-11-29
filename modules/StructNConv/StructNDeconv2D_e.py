
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F


class StructNDeconv2D_e(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, in_channels=1, out_channels=1, groups=1,
                 kernel_size=1, stride=1, padding=0, dilation=1):
        super(StructNDeconv2D_e, self).__init__()

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

        # Define Parameters
        if mirror_weights:
            self.spatial_weight0 = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size, self.kernel_size))
            self.spatial_weight1 = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
            self.spatial_weight3 = torch.nn.Parameter(data=torch.Tensor(self.in_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            self.spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.in_channels * 4, 1, self.kernel_size, self.kernel_size))
        
        # Init Parameters
        if self.init_method == 'x':  # Xavier
            if mirror_weights:
               torch.nn.init.xavier_uniform_(self.spatial_weight0) + 1
               torch.nn.init.xavier_uniform_(self.spatial_weight1) + 1
               torch.nn.init.xavier_uniform_(self.spatial_weight3) + 1
            else:
               torch.nn.init.xavier_uniform_(self.spatial_weight) + 1
        else:  # elif self.init_method == 'k': # Kaiming
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


    def forward(self, e, ce, target_shape):
        if self.mirror_weights:
            self.spatial_weight = torch.cat((self.spatial_weight0,
                                             torch.cat((self.spatial_weight1, self.spatial_weight1[:,:,:,:-1].flip(dims=(3,))), dim=3),
                                             self.spatial_weight0.flip(dims=(3,)),
                                             torch.cat((self.spatial_weight3, self.spatial_weight3[:,:,:,:-1].flip(dims=(3,))), dim=3)), dim=0)
        else:
            channel_weight = self.channel_weight
                
        shape = e.shape
        e, ce = e.view(shape[0],shape[1] * shape[2],shape[3], shape[4]), ce.view(shape[0],shape[1] * shape[2],shape[3], shape[4])
        
        output_padding = (target_shape[3] - ((shape[3] - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1),
                          target_shape[4] - ((shape[4] - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1))

        # Normalized Convolution along spatial dimensions
        nom = F.conv_transpose2d(ce * e, self.spatial_weight, groups=self.in_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=output_padding).squeeze(2)
        denom = F.conv_transpose2d(ce, self.spatial_weight, groups=self.in_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=output_padding).squeeze(2)
        cdenom = F.conv_transpose2d(torch.ones_like(ce), self.spatial_weight, groups=self.in_channels * 4, stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=output_padding).squeeze(2)
        e = nom / (denom + self.eps)
        ce = denom / (cdenom + self.eps)
        
        shape = e.shape
        return e.view(shape[0],shape[1] // 4, 4,shape[2], shape[3]), ce.view(shape[0],shape[1] // 4, 4,shape[2], shape[3])
