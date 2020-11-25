########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch

#from modules.StructNConv.s_prod_KernelChannels import s_prod_KernelChannels
from modules.StructNConv.e_prod_KernelChannels3 import e_prod_KernelChannels3
from modules.StructNConv.e_prod_KernelChannels5 import e_prod_KernelChannels5

from modules.StructNConv.StructNMaxPool2D_e import StructNMaxPool2D_e
from modules.StructNConv.StructNMaxPool2D_d_with_e import StructNMaxPool2D_d_with_e

from modules.StructNConv.StructNConv2D_e_with_d import StructNConv2D_e_with_d
from modules.StructNConv.StructNConv2D_d_with_s import StructNConv2D_d_with_s
from modules.StructNConv.StructNConv2D_d import StructNConv2D_d
from modules.StructNConv.StructNConv2D_out import StructNConv2D_out

from modules.StructNConv.StructNDeconv2D import StructNDeconv2D
from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample
from modules.StructNConv.NearestNeighbourUpsample2 import NearestNeighbourUpsample2
from modules.StructNConv.ReturnNone import ReturnNone


class CNN(torch.nn.Module):

    def __init__(self, params):
        num_channels = params['num_channels']
        devalue_pooled_confidence = params['devalue_pooled_confidence']

        maxpool_s = params['maxpool_s']
        nn_upsample_s = params['nn_upsample_s']

        maxpool_d = params['maxpool_d']
        nn_upsample_d = params['nn_upsample_d']

        assert params['lidar_padding'] == 0
        super().__init__()
        
        # boundary/smoothness modules
        self.nconv_s = torch.nn.ModuleList([StructNConv2D_e_with_d(in_channels=1, out_channels=num_channels,init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        # pool
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1)])
        # pooling
        if maxpool_s:
            self.npool_s = StructNMaxPool2D_e(channels=num_channels,
                                              kernel_size=2, stride=2, padding=0,
                                              devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_s = StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=2, stride=2, padding=0, dilation=1,
                                                  devalue_pooled_confidence=devalue_pooled_confidence)
        if nn_upsample_s:
            self.nup_s = NearestNeighbourUpsample2(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_s = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                         kernel_size=2, stride=2, padding=0, dilation=1)


        # s_prod
        self.s_prod_1 = e_prod_KernelChannels5()
        if maxpool_d:
            self.s_prod_pool = ReturnNone()
        else:
            self.s_prod_pool = e_prod_KernelChannels(kernel_size=2, stride=2, padding=0, dilation=1)
        self.s_prod_2 = e_prod_KernelChannels3()


        # depth modules
        # in_channels not 1 because of multiplication with output of nconv1_s
        self.nconv_d = torch.nn.ModuleList([StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        # pool
                        StructNConv2D_d_with_s(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_d_with_s(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_d_with_s(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        # out
                        StructNConv2D_out(in_channels=num_channels, init_method=params['init_method'])])
        # pooling
        if maxpool_d:
            self.npool_d = StructNMaxPool2D_d_with_e(kernel_size=2, stride=2, padding=0)
        else:
            self.npool_d = StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=2, stride=2, padding=0, dilation=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                         kernel_size=2, stride=2, padding=0, dilation=1)
        self.outs = ['d', 'cd', 'e', 'ce']

        self.enforce_limits()

    def enforce_limits(self):
        for nconv in self.nconv_d:
            nconv.enforce_limits()
        for nconv in self.nconv_s:
            nconv.enforce_limits()
        self.npool_d.enforce_limits()
        self.nup_d.enforce_limits()
        self.npool_s.enforce_limits()
        self.nup_s.enforce_limits()
    
    
    def forward(self, d_0, cd_0, s_0=None, cs_0 = None):
        assert d_0.shape[2] % (self.nup_d.kernel_size ** 3) == 0
        assert d_0.shape[3] % (self.nup_d.kernel_size ** 3) == 0
        assert d_0.shape == cd_0.shape

        if s_0 is None:
            s_0 = cs_0 = torch.zeros(size=(d_0.shape[0], d_0.shape[1], 4, d_0.shape[2], d_0.shape[3]), device=d_0.device)

        # Stage 0
        s_0, cs_0 = self.nconv_s[0](d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_1(s_0, cs_0)
        d_0, cd_0 = self.nconv_d[0](d_0, cd_0, s_0, cs_0, s_prod)
        s_0, cs_0 = self.nconv_s[1](d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_1(s_0, cs_0)
        d_0, cd_0 = self.nconv_d[1](d_0, cd_0, s_0, cs_0, s_prod)
        s_0, cs_0 = self.nconv_s[2](d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_1(s_0, cs_0)
        d_0, cd_0 = self.nconv_d[2](d_0, cd_0, s_0, cs_0, s_prod)

        # Stage 1
        s_1, cs_1 = self.npool_s(d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_pool(s_0, cs_0)
        d_1, cd_1 = self.npool_d(d_0, cd_0, s_0, cs_0, s_prod)
        s_1, cs_1 = self.nconv_s[1](d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_1(s_1, cs_1)
        d_1, cd_1 = self.nconv_d[1](d_1, cd_1, s_1, cs_1, s_prod)
        s_1, cs_1 = self.nconv_s[2](d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_1(s_1, cs_1)
        d_1, cd_1 = self.nconv_d[2](d_1, cd_1,  s_1, cs_1, s_prod)

        # Stage 2
        s_2, cs_2 = self.npool_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_pool(s_1, cs_1)
        d_2, cd_2 = self.npool_d(d_1, cd_1, s_1, cs_1, s_prod)
        s_2, cs_2 = self.nconv_s[1](d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_1(s_2, cs_2)
        d_2, cd_2 = self.nconv_d[1](d_2, cd_2, s_2, cs_2, s_prod)

        # Stage 3
        s_3, cs_3 = self.npool_s(d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_pool(s_2, cs_2)
        d_3, cd_3 = self.npool_d(d_2, cd_2, s_2, cs_2, s_prod)
        s_3, cs_3 = self.nconv_s[1](d_3, cd_3, s_3, cs_3)
        s_prod = self.s_prod_1(s_3, cs_3)
        d_3, cd_3 = self.nconv_d[1](d_3, cd_3, s_3, cs_3, s_prod)

        # Stage 2
        s_32, cs_32 = self.nup_s(s_3, cs_3)
        d_32, cd_32 = self.nup_d(d_3, cd_3)
        s_2, cs_2 = torch.cat((s_32, s_2), 1), torch.cat((cs_32, cs_2), 1)
        d_2, cd_2 = torch.cat((d_32, d_2), 1), torch.cat((cd_32, cd_2), 1)
        s_2, cs_2 = self.nconv_s[3](d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_2(s_2, cs_2).repeat(1, 2, 1, 1, 1)
        d_2, cd_2 = self.nconv_d[3](d_2, cd_2, s_2, cs_2, s_prod)

        # Stage 1
        s_21, cs_21 = self.nup_s(s_2, cs_2)
        d_21, cd_21 = self.nup_d(d_2, cd_2)
        s_1, cs_1 = torch.cat((s_21, s_1), 1), torch.cat((cs_21, cs_1), 1)
        d_1, cd_1 = torch.cat((d_21, d_1), 1), torch.cat((cd_21, cd_1), 1)
        s_1, cs_1 = self.nconv_s[4](d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_2(s_1, cs_1).repeat(1, 2, 1, 1, 1)
        d_1, cd_1 = self.nconv_d[4](d_1, cd_1, s_1, cs_1, s_prod)

        # Stage 0
        s_10, cs_10 = self.nup_s(s_1, cs_1)
        d_10, cd_10 = self.nup_d(d_1, cd_1)
        s_0, cs_0 = torch.cat((s_10, s_0), 1), torch.cat((cs_10, cs_0), 1)
        d_0, cd_0 = torch.cat((d_10, d_0), 1), torch.cat((cd_10, cd_0), 1)
        s_0, cs_0 = self.nconv_s[5](d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_2(s_0, cs_0).repeat(1, 2, 1, 1, 1)
        d_0, cd_0 = self.nconv_d[5](d_0, cd_0, s_0, cs_0, s_prod)

        # output
        d, cd = self.nconv_d[6](d_0, cd_0)
        s, cs = self.nconv_d[6](s_0.view(d.shape[0], -1, d.shape[2]*4, d.shape[3]), cs_0.view(d.shape[0], -1, d.shape[2]*4, d.shape[3]))
        s[cs == 0] = 0
        return d, cd, s.view(d.shape[0], 4, d.shape[2], d.shape[3]), cs.view(d.shape[0], 4, d.shape[2], d.shape[3])
