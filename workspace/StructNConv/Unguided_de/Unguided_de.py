########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch

from modules.StructNConv.StructNMaxPool2D_e import StructNMaxPool2D_e
from modules.StructNConv.StructNMaxPool2D_d_with_e import StructNMaxPool2D_d_with_e

from modules.StructNConv.StructNConv2D_e_with_d import StructNConv2D_e_with_d
from modules.StructNConv.StructNConv2D_d_with_e import StructNConv2D_d_with_e
from modules.StructNConv.StructNConv2D_d import StructNConv2D_d
from modules.StructNConv.StructNConv2D_out import StructNConv2D_out

from modules.StructNConv.StructNDeconv2D import StructNDeconv2D
from modules.StructNConv.StructNDeconv2D_e import StructNDeconv2D_e
from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample
from modules.StructNConv.NearestNeighbourUpsample2 import NearestNeighbourUpsample2
from modules.StructNConv.ReturnNone import ReturnNone


class CNN(torch.nn.Module):

    def __init__(self, params):
        num_channels = params['num_channels']
        devalue_pooled_confidence = params['devalue_pooled_confidence']

        maxpool_e = params['maxpool_e']
        nn_upsample_e = params['nn_upsample_e']

        maxpool_d = params['maxpool_d']
        nn_upsample_d = params['nn_upsample_d']

        assert params['lidar_padding'] == 0
        super().__init__()
        
        # boundary/smoothness modules
        self.nconv_e = torch.nn.ModuleList([StructNConv2D_e_with_d(in_channels=1, out_channels=num_channels,init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        # pool
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1)])
        # pooling
        if maxpool_e:
            self.npool_e = StructNMaxPool2D_e(channels=num_channels,
                                              kernel_size=3, stride=2, padding=1,
                                              devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_e = StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1,
                                                  devalue_pooled_confidence=devalue_pooled_confidence)
        if nn_upsample_e:
            self.nup_e = NearestNeighbourUpsample2(kernel_size=3, stride=2, padding=1)
        else:
            self.nup_e = StructNDeconv2D_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1)


        # depth modules
        # in_channels not 1 because of multiplication with output of nconv1_e
        self.nconv_d = torch.nn.ModuleList([StructNConv2D_d_with_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        StructNConv2D_d_with_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        StructNConv2D_d_with_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2),
                        # pool
                        StructNConv2D_d_with_e(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1),
                        StructNConv2D_d_with_e(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1),
                        StructNConv2D_d_with_e(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1),
                        # out
                        StructNConv2D_out(in_channels=num_channels, init_method=params['init_method'])])
        # pooling
        if maxpool_d:
            self.npool_d = StructNMaxPool2D_d_with_e(kernel_size=3, stride=2, padding=1)
        else:
            self.npool_d = StructNConv2D_d_with_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=3, stride=2, padding=1)
        else:
            self.nup_d = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1)
        self.outs = ['d', 'cd', 'e', 'ce']

        self.enforce_limits()

    def enforce_limits(self):
        for nconv in self.nconv_d:
            nconv.enforce_limits()
        for nconv in self.nconv_e:
            nconv.enforce_limits()
        self.npool_d.enforce_limits()
        self.nup_d.enforce_limits()
        self.npool_e.enforce_limits()
        self.nup_e.enforce_limits()
    
    
    def forward(self, d_0, cd_0, e_0=None, ce_0 = None):
        if e_0 is None:
            e_0 = ce_0 = torch.zeros(size=(d_0.shape[0], d_0.shape[1], 4, d_0.shape[2], d_0.shape[3]), device=d_0.device)        

        # Stage 0
        e_0, ce_0 = self.nconv_e[0](d_0, cd_0, e_0, ce_0)
        d_0, cd_0 = self.nconv_d[0](d_0, cd_0, e_0, ce_0)
        e_0, ce_0 = self.nconv_e[1](d_0, cd_0, e_0, ce_0)
        d_0, cd_0 = self.nconv_d[1](d_0, cd_0, e_0, ce_0)
        e_0, ce_0 = self.nconv_e[2](d_0, cd_0, e_0, ce_0)
        d_0, cd_0 = self.nconv_d[2](d_0, cd_0, e_0, ce_0)

        # Stage 1
        e_1, ce_1 = self.npool_e(d_0, cd_0, e_0, ce_0)
        d_1, cd_1 = self.npool_d(d_0, cd_0, e_0, ce_0)
        e_1, ce_1 = self.nconv_e[1](d_1, cd_1, e_1, ce_1)
        d_1, cd_1 = self.nconv_d[1](d_1, cd_1, e_1, ce_1)
        e_1, ce_1 = self.nconv_e[2](d_1, cd_1, e_1, ce_1)
        d_1, cd_1 = self.nconv_d[2](d_1, cd_1,  e_1, ce_1)

        # Stage 2
        e_2, ce_2 = self.npool_e(d_1, cd_1, e_1, ce_1)
        d_2, cd_2 = self.npool_d(d_1, cd_1, e_1, ce_1)
        e_2, ce_2 = self.nconv_e[1](d_2, cd_2, e_2, ce_2)
        d_2, cd_2 = self.nconv_d[1](d_2, cd_2, e_2, ce_2)

        # Stage 3
        e_3, ce_3 = self.npool_e(d_2, cd_2, e_2, ce_2)
        d_3, cd_3 = self.npool_d(d_2, cd_2, e_2, ce_2)
        e_3, ce_3 = self.nconv_e[1](d_3, cd_3, e_3, ce_3)
        d_3, cd_3 = self.nconv_d[1](d_3, cd_3, e_3, ce_3)

        # Stage 2
        e_32, ce_32 = self.nup_e(e_3, ce_3, e_2.shape)
        d_32, cd_32 = self.nup_d(d_3, cd_3, d_2.shape)
        e_2, ce_2 = torch.cat((e_32, e_2), 1), torch.cat((ce_32, ce_2), 1)
        d_2, cd_2 = torch.cat((d_32, d_2), 1), torch.cat((cd_32, cd_2), 1)
        e_2, ce_2 = self.nconv_e[3](d_2, cd_2, e_2, ce_2)
        d_2, cd_2 = self.nconv_d[3](d_2, cd_2, e_2, ce_2)

        # Stage 1
        e_21, ce_21 = self.nup_e(e_2, ce_2, e_1.shape)
        d_21, cd_21 = self.nup_d(d_2, cd_2, d_1.shape)
        e_1, ce_1 = torch.cat((e_21, e_1), 1), torch.cat((ce_21, ce_1), 1)
        d_1, cd_1 = torch.cat((d_21, d_1), 1), torch.cat((cd_21, cd_1), 1)
        e_1, ce_1 = self.nconv_e[4](d_1, cd_1, e_1, ce_1)
        d_1, cd_1 = self.nconv_d[4](d_1, cd_1, e_1, ce_1)

        # Stage 0
        e_10, ce_10 = self.nup_e(e_1, ce_1, e_0.shape)
        d_10, cd_10 = self.nup_d(d_1, cd_1, d_0.shape)
        e_0, ce_0 = torch.cat((e_10, e_0), 1), torch.cat((ce_10, ce_0), 1)
        d_0, cd_0 = torch.cat((d_10, d_0), 1), torch.cat((cd_10, cd_0), 1)
        e_0, ce_0 = self.nconv_e[5](d_0, cd_0, e_0, ce_0)
        d_0, cd_0 = self.nconv_d[5](d_0, cd_0, e_0, ce_0)

        # output
        d, cd = self.nconv_d[6](d_0, cd_0)
        s, cs = self.nconv_d[6](e_0.view(d.shape[0], -1, d.shape[2] * 4, d.shape[3]), ce_0.view(d.shape[0], -1, d.shape[2] * 4, d.shape[3]))
        s[cs == 0] = 0
        return d, cd, s.view(d.shape[0], 4, d.shape[2], d.shape[3]), cs.view(d.shape[0], 4, d.shape[2], d.shape[3])
