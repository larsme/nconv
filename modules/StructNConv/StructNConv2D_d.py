
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
        if mirror_weights == 2:
            rad = (self.kernel_size + 1) // 2
            ntri = (rad * (rad + 1)) // 2
            ntri -=1
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, ntri))
        elif mirror_weights:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, (self.kernel_size + 1) // 2))
        else:
            spatial_weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, 1, self.kernel_size, self.kernel_size))

        # Init Parameters
        if 'x' in self.init_method:  # Xavier
            if self.in_channels > 1:
                torch.nn.init.xavier_uniform_(self.channel_weight) 
            torch.nn.init.xavier_uniform_(spatial_weight) 
        elif 'k' in self.init_method: # Kaiming
            if self.in_channels > 1:
                torch.nn.init.kaiming_uniform_(self.channel_weight)
            torch.nn.init.kaiming_uniform_(spatial_weight)
        
        if mirror_weights:
            self.true_spatial_weight = spatial_weight
        else:
            self.spatial_weight = spatial_weight
            
    def print(self, s_list):
        spatial_weight, channel_weight = self.prepare_weights()

        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                s_list[y] += '{0:0.2f}, '.format(spatial_weight[:,:,y,x].item())
        return s_list

    def enforce_limits(self):
        # Enforce positive weights
        if self.mirror_weights:
            self.true_spatial_weight.data = F.softplus(self.true_spatial_weight, beta=10)
        else:
            self.spatial_weight.data = F.softplus(self.spatial_weight, beta=10)
        if self.in_channels > 1:
            self.channel_weight.data = F.softplus(self.channel_weight, beta=10)

    def prepare_weights(self):
        if self.mirror_weights == 2:
            if self.kernel_size == 5:
                spatial_weight = torch.stack((self.true_spatial_weight[:,0],self.true_spatial_weight[:,1], self.true_spatial_weight[:,2], self.true_spatial_weight[:,1], self.true_spatial_weight[:,0],
                                                 self.true_spatial_weight[:,1],self.true_spatial_weight[:,3], self.true_spatial_weight[:,4], self.true_spatial_weight[:,3], self.true_spatial_weight[:,1],
                                                 self.true_spatial_weight[:,2],self.true_spatial_weight[:,4], 5 * torch.ones_like(self.true_spatial_weight[:,2]), self.true_spatial_weight[:,4], self.true_spatial_weight[:,2],
                                                 self.true_spatial_weight[:,1],self.true_spatial_weight[:,3], self.true_spatial_weight[:,4], self.true_spatial_weight[:,3], self.true_spatial_weight[:,1],
                                                 self.true_spatial_weight[:,0],self.true_spatial_weight[:,1], self.true_spatial_weight[:,2], self.true_spatial_weight[:,1], self.true_spatial_weight[:,0])
                                                ,dim=1).view(self.out_channels, 1, self.kernel_size, self.kernel_size)
            elif self.kernel_size == 3:
                spatial_weight = torch.stack((self.true_spatial_weight[:,0],self.true_spatial_weight[:,1], self.true_spatial_weight[:,0],
                                                 self.true_spatial_weight[:,1],5 * torch.ones_like(self.true_spatial_weight[:,1]), self.true_spatial_weight[:,1],
                                                 self.true_spatial_weight[:,0],self.true_spatial_weight[:,1], self.true_spatial_weight[:,0])
                                                 ,dim=1).view(self.out_channels, 1, self.kernel_size, self.kernel_size)
        elif self.mirror_weights:
            spatial_weight = torch.cat((self.true_spatial_weight, self.true_spatial_weight[:,:,:,:-1].flip(dims=(3,))), dim=3)
        else:
            spatial_weight=self.spatial_weight
        spatial_weight = spatial_weight / spatial_weight.sum(dim=[2,3], keepdim=True)
        
        if self.in_channels > 1:
            channel_weight = self.channel_weight / self.channel_weight.sum(dim=1, keepdim=True)
        else:
            channel_weight = None

        return spatial_weight, channel_weight

    def forward(self, d, cd):
        spatial_weight, channel_weight = self.prepare_weights()

        # Normalized Convolution along channel dimensions
        if self.in_channels > 1:
            nom = F.conv2d(cd * d, channel_weight, groups=self.groups)
            cd = F.conv2d(cd, channel_weight, groups=self.groups)
            d = nom / (cd + self.eps)
        elif self.out_channels > 1:
            d, cd = d.expand(-1, self.out_channels,-1,-1), cd.expand(-1, self.out_channels,-1,-1)

        # Normalized Convolution along spatial dimensions
        nom = F.conv2d(cd * d, spatial_weight, groups=self.out_channels, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)
        cd = F.conv2d(cd, spatial_weight, groups=self.out_channels, stride=self.stride,
                            padding=self.padding, dilation=self.dilation)
        d = nom / (cd + self.eps)

        if self.devalue_conf != 1:
            return d, cd * self.devalue_conf
        else:
            return d, cd
