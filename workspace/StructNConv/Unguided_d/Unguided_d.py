########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch

from modules.StructNConv.StructNConv2D_d import StructNConv2D_d
from modules.StructNConv.StructNMaxPool2D_d import StructNMaxPool2D_d
from modules.StructNConv.StructNDeconv2D import StructNDeconv2D
from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample


class CNN(torch.nn.Module):

    def __init__(self, params):
        pos_fn = params['enforce_pos_weights']
        num_channels = params['num_channels']
        maxpool_d = params['maxpool_d']
        nn_upsample_d = params['nn_upsample_d']
        use_conv_bias_d = params['use_conv_bias_d']
        use_deconv_bias_d = params['use_deconv_bias_d']
        const_bias_init_d = params['const_bias_init_d']
        devalue_pooled_confidence = params['devalue_pooled_confidence']
        super().__init__() 
        
        self.pos_fn = pos_fn

        self.lidar_padding = params['lidar_padding']
        # If sparse input depth maps are calculated from a point cloud,
        # it is possible to use an input map larger than the image
        # and pad with real point measurements instead of zero padding
        self.l_pad_2 = 2*2  # lidar padding on stage 3 upsampled to stage 2
        self.l_pad_1 = (2 + self.l_pad_2) * 2  # lidar padding on stage 2 upsampled to stage 1
        self.l_pad_0 = (2 + 2 + self.l_pad_1) * 2  # lidar padding on stage 1 upsampled to stage 0
        l_pad = 2 + 2 + 2 + self.l_pad_0  # lidar padding of entire structure
        # (don't replace one paddings for convenience)
        assert self.lidar_padding == 0\
            or self.lidar_padding == l_pad
        if not self.lidar_padding:
            self.l_pad_0 = self.l_pad_1 = self.l_pad_2 = 0

        # depth modules
        self.nconv1_d = StructNConv2D_d(in_channels=1, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=0 if self.lidar_padding else 2, dilation=1)
        self.nconv2_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=0 if self.lidar_padding else 2, dilation=1)
        self.nconv3_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=5, stride=1, padding=0 if self.lidar_padding else 2, dilation=1)
        if maxpool_d:
            self.npool_d = StructNMaxPool2D_d(kernel_size=2, stride=2, padding=0,
                                              devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels, channel_first=False,
                                           pos_fn=pos_fn, init_method=params['init_method'],
                                           use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                           kernel_size=2, stride=2, padding=0, dilation=1,
                                           devalue_pooled_confidence=devalue_pooled_confidence)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels,
                                         pos_fn=pos_fn, init_method=params['init_method'], use_bias=use_deconv_bias_d,
                                         kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_d = StructNConv2D_d(in_channels=2 * num_channels, out_channels=num_channels, channel_first=True,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_d = StructNConv2D_d(in_channels=num_channels, out_channels=1,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=1, stride=1, padding=0, dilation=1)


    def forward(self, d_0, cd_0):
        assert d_0.shape == cd_0.shape

        # Stage 0
        d_0, cd_0 = self.nconv1_d(d_0, cd_0)
        d_0, cd_0 = self.nconv2_d(d_0, cd_0)
        d_0, cd_0 = self.nconv3_d(d_0, cd_0)

        # Stage 1
        assert d_0.shape[2] % 2 == 0
        assert d_0.shape[3] % 2 == 0
        d_1, cd_1 = self.npool_d(d_0, cd_0)
        d_1, cd_1 = self.nconv2_d(d_1, cd_1)
        d_1, cd_1 = self.nconv3_d(d_1, cd_1)

        # Stage 2
        assert d_1.shape[2] % 2 == 0
        assert d_1.shape[3] % 2 == 0
        d_2, cd_2 = self.npool_d(d_1, cd_1)
        d_2, cd_2 = self.nconv2_d(d_2, cd_2)

        # Stage 3
        assert d_2.shape[2] % 2 == 0
        assert d_2.shape[3] % 2 == 0
        d_3, cd_3 = self.npool_d(d_2, cd_2)
        d_3, cd_3 = self.nconv2_d(d_3, cd_3)

        # Stage 2
        d_32, cd_32 = self.nup_d(d_3, cd_3)
        d_2, cd_2 = self.nconv4_d(torch.cat((d_32, d_2[:, :, self.l_pad_2: d_2.shape[2]-self.l_pad_2,
                                                       self.l_pad_2: d_2.shape[3]-self.l_pad_2]), 1),
                                  torch.cat((cd_32, cd_2[:, :, self.l_pad_2: cd_2.shape[2]-self.l_pad_2,
                                                         self.l_pad_2: cd_2.shape[3]-self.l_pad_2]), 1))
        
        # Stage 1
        d_21, cd_21 = self.nup_d(d_2, cd_2)
        d_1, cd_1 = self.nconv5_d(torch.cat((d_21, d_1[:, :, self.l_pad_1: d_1.shape[2]-self.l_pad_1,
                                                       self.l_pad_1: d_1.shape[3]-self.l_pad_1]), 1),
                                  torch.cat((cd_21, cd_1[:, :, self.l_pad_1: cd_1.shape[2]-self.l_pad_1,
                                                         self.l_pad_1: cd_1.shape[3]-self.l_pad_1]), 1))

        # Stage 0
        d_10, cd_10 = self.nup_d(d_1, cd_1)
        d_0, cd_0 = self.nconv6_d(torch.cat((d_10, d_0[:, :, self.l_pad_0: d_0.shape[2]-self.l_pad_0,
                                                       self.l_pad_0: d_0.shape[3]-self.l_pad_0]), 1),
                                  torch.cat((cd_10, cd_0[:, :, self.l_pad_0: cd_0.shape[2]-self.l_pad_0,
                                                         self.l_pad_0: cd_0.shape[3]-self.l_pad_0]), 1))
        
        # output
        d, cd = self.nconv7_d(d_0, cd_0)
        return d, cd
