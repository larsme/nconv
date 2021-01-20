
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F


class StructNDeconv2D(torch.nn.Module):
    def __init__(self, init_method='k', mirror_weights=False, in_channels=1, out_channels=1, groups=1,
                 kernel_size=1, stride=1, padding=0, dilation=1):
        super(StructNDeconv2D, self).__init__()

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
            self.spatial_weight = torch.nn.Parameter(data=torch.ones(self.in_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            self.spatial_weight = torch.nn.Parameter(data=torch.ones(self.in_channels, 1, self.kernel_size, self.kernel_size))

        # Init Parameters
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.spatial_weight)
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.spatial_weight)

    def prepare_weights(self):
        if self.mirror_weights:
            spatial_weight = F.softplus(self.spatial_weight)
            spatial_weight = torch.cat((spatial_weight, spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)
        else:
            spatial_weight = F.softplus(self.spatial_weight)
        ones = torch.ones((1, self.in_channels, 3*self.kernel_size-2, 3*self.kernel_size-2), device=spatial_weight.device)
        spatial_weight = spatial_weight / F.conv2d(ones, spatial_weight, stride=self.stride, padding=0, groups=self.in_channels).view(self.in_channels, 1,self.kernel_size,self.kernel_size)

        return spatial_weight
    
    def prep_eval(self):
        self.weights = self.prepare_weights()
        
    def forward(self, d, cd, target_shape):
        if self.training:
            spatial_weight = self.prepare_weights()
        else:
            spatial_weight = self.weights

        shape = d.shape
        output_padding = (target_shape[2] - ((shape[2] - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1),
                          target_shape[3] - ((shape[3] - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1))

        # Normalized Deconvolution along spatial dimensions
        nom = F.conv_transpose2d(cd * d, spatial_weight, groups=self.in_channels,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=output_padding)
        cd = F.conv_transpose2d(cd, spatial_weight, groups=self.in_channels,
                                stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=output_padding)
        d = nom / (cd + self.eps)

        return d, cd
