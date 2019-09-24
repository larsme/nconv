########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch

from modules.StructNConv.s_prod_KernelChannels import s_prod_KernelChannels

from modules.StructNConv.StructNMaxPool2D_s import StructNMaxPool2D_s
from modules.StructNConv.StructNMaxPool2D_dg_with_s import StructNMaxPool2D_dg_with_s
from modules.StructNConv.StructNMaxPool2D_d import StructNMaxPool2D_d

from modules.StructNConv.StructNConv2D_s_with_d import StructNConv2D_s_with_d
from modules.StructNConv.StructNConv2D_dg_with_s import StructNConv2D_dg_with_s
from modules.StructNConv.StructNConv2D_d_with_sg import StructNConv2D_d_with_sg
from modules.StructNConv.StructNConv2D_g_with_ds import StructNConv2D_g_with_ds
from modules.StructNConv.StructNConv2D_d import StructNConv2D_d

from modules.StructNConv.StructNDeconv2D import StructNDeconv2D
from modules.StructNConv.NearestNeighbourUpsample import NearestNeighbourUpsample
from modules.StructNConv.ReturnNone import ReturnNone


class CNN(torch.nn.Module):

    def __init__(self, params):
        pos_fn = params['enforce_pos_weights']
        num_channels = params['num_channels']
        devalue_pooled_confidence = params['devalue_pooled_confidence']

        maxpool_s = params['maxpool_s']
        nn_upsample_s = params['nn_upsample_s']
        use_conv_bias_s = params['use_conv_bias_s']
        const_bias_init_s = params['const_bias_init_s']
        use_deconv_bias_s = params['use_deconv_bias_s']

        maxpool_dg = params['maxpool_dg']

        nn_upsample_g = params['nn_upsample_g']
        use_conv_bias_g = params['use_conv_bias_g']
        const_bias_init_g = params['const_bias_init_g']
        use_deconv_bias_g = params['use_deconv_bias_g']

        nn_upsample_d = params['nn_upsample_d']
        use_conv_bias_d = params['use_conv_bias_d']
        const_bias_init_d = params['const_bias_init_d']
        use_deconv_bias_d = params['use_deconv_bias_d']

        assert params['lidar_padding'] == 0
        super().__init__()

        self.pos_fn = pos_fn

        # boundary/smoothness modules
        self.nconv1_s = StructNConv2D_s_with_d(in_channels=1, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_s = StructNConv2D_s_with_d(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_s = StructNConv2D_s_with_d(in_channels=num_channels, out_channels=num_channels,
                                               channel_first=False,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=5, stride=1, padding=2, dilation=1)
        # pooling
        if maxpool_s:
            self.npool_s = StructNMaxPool2D_s(channels=num_channels,
                                              kernel_size=2, stride=2, padding=0,
                                              pos_fn=pos_fn, init_method=params['init_method'],
                                              devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_s = StructNConv2D_s_with_d(in_channels=num_channels, out_channels=num_channels,
                                                  channel_first=False,
                                                  pos_fn=pos_fn, init_method=params['init_method'],
                                                  use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                                  kernel_size=2, stride=2, padding=0, dilation=1,
                                                  devalue_pooled_confidence=devalue_pooled_confidence)
        if nn_upsample_s:
            self.nup_s = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_s = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels,
                                         pos_fn=pos_fn, init_method=params['init_method'],
                                         use_bias=use_deconv_bias_s,
                                         kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_s = StructNConv2D_s_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_s = StructNConv2D_s_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_s = StructNConv2D_s_with_d(in_channels=2 * num_channels, out_channels=num_channels,
                                               channel_first=True,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_s = StructNConv2D_s_with_d(in_channels=num_channels, out_channels=1,
                                               pos_fn=pos_fn, init_method=params['init_method'],
                                               use_bias=use_conv_bias_s, const_bias_init=const_bias_init_s,
                                               kernel_size=1, stride=1, padding=0, dilation=1)

        # s_prod
        self.s_prod_1 = s_prod_KernelChannels(kernel_size=5, stride=1, padding=2, dilation=1)
        if maxpool_dg:
            self.s_prod_pool = ReturnNone()
        else:
            self.s_prod_pool = s_prod_KernelChannels(kernel_size=2, stride=2, padding=0, dilation=1)
        self.s_prod_2 = s_prod_KernelChannels(kernel_size=3, stride=1, padding=1, dilation=1)

        # shared pooling
        if maxpool_dg:
            self.npool_dg = StructNMaxPool2D_dg_with_s(channels=num_channels,
                                                       kernel_size=2, stride=2, padding=0,
                                                       pos_fn=pos_fn, init_method=params['init_method'],
                                                       devalue_pooled_confidence=devalue_pooled_confidence)
            self.npool_d = StructNMaxPool2D_d( kernel_size=2, stride=2, padding=0,
                                               devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_dg = StructNConv2D_dg_with_s(in_channels=num_channels, out_channels=num_channels,
                                                    channel_first=False,
                                                    pos_fn=pos_fn, init_method=params['init_method'],
                                                    use_bias_d=use_conv_bias_d, const_bias_init_d=const_bias_init_d,
                                                    use_bias_g=use_conv_bias_g, const_bias_init_g=const_bias_init_g,
                                                    kernel_size=2, stride=2, padding=0, dilation=1,
                                                    devalue_pooled_confidence=devalue_pooled_confidence)
            self.npool_d = StructNConv2D_d(in_channels=num_channels, out_channels=num_channels,
                                           channel_first=False,
                                           pos_fn=pos_fn, init_method=params['init_method'],
                                           use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                           kernel_size=2, stride=2, padding=0, dilation=1,
                                           devalue_pooled_confidence=devalue_pooled_confidence)

        # gradient modules
        # in_channels not 1 because of multiplication with output of nconv1_s
        self.nconv1_g = StructNConv2D_g_with_ds(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_g = StructNConv2D_g_with_ds(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_g = StructNConv2D_g_with_ds(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        if nn_upsample_g:
            self.nup_gx = self.nup_gy = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_gx = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels,
                                          pos_fn=pos_fn, init_method=params['init_method'],
                                          use_bias=use_deconv_bias_g,
                                          kernel_size=2, stride=2, padding=0, dilation=1)
            self.nup_gy = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels,
                                          pos_fn=pos_fn, init_method=params['init_method'],
                                          use_bias=use_deconv_bias_g,
                                          kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_g = StructNConv2D_g_with_ds(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_g = StructNConv2D_g_with_ds(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_g = StructNConv2D_g_with_ds(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=3, stride=1, padding=1, dilation=1)

        self.nconv7_g = StructNConv2D_g_with_ds(in_channels=num_channels, out_channels=1,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_g, const_bias_init=const_bias_init_g,
                                                kernel_size=1, stride=1, padding=0, dilation=1)

        # depth modules
        # in_channels not 1 because of multiplication with output of nconv1_s and addition with output of nconv1_g
        self.nconv1_d = StructNConv2D_d_with_sg(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv2_d = StructNConv2D_d_with_sg(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        self.nconv3_d = StructNConv2D_d_with_sg(in_channels=num_channels, out_channels=num_channels,
                                                channel_first=False,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=5, stride=1, padding=2, dilation=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=2, stride=2, padding=0)
        else:
            self.nup_d = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels,
                                         pos_fn=pos_fn, init_method=params['init_method'],
                                         use_bias=use_deconv_bias_d,
                                         kernel_size=2, stride=2, padding=0, dilation=1)

        self.nconv4_d = StructNConv2D_d_with_sg(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv5_d = StructNConv2D_d_with_sg(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=3, stride=1, padding=1, dilation=1)
        self.nconv6_d = StructNConv2D_d_with_sg(in_channels=2 * num_channels, out_channels=num_channels,
                                                channel_first=True,
                                                pos_fn=pos_fn, init_method=params['init_method'],
                                                use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                                kernel_size=3, stride=1, padding=1, dilation=1)

        # no StructNConv2D_d_with_g because of kernel size 1
        self.nconv7_d = StructNConv2D_d(in_channels=num_channels, out_channels=1,
                                        pos_fn=pos_fn, init_method=params['init_method'],
                                        use_bias=use_conv_bias_d, const_bias_init=const_bias_init_d,
                                        kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, d_0, cd_0):
        d_1, cd_1 = self.npool_d(d_0, cd_0)

        assert d_1.shape[2] % (self.nup_d.kernel_size**3) == 0
        assert d_1.shape[3] % (self.nup_d.kernel_size**3) == 0
        assert d_1.shape == cd_1.shape

        s_1 = cs_1 = torch.zeros_like(cd_1)
        gx_1 = cgx_1 = gy_1 = cgy_1 = torch.zeros_like(cd_1)

        # Stage 1
        s_1, cs_1 = self.nconv1_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_1(s_1, cs_1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv1_g(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        d_1, cd_1 = self.nconv1_d(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        s_1, cs_1 = self.nconv2_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_1(s_1, cs_1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv2_g(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        d_1, cd_1 = self.nconv2_d(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        s_1, cs_1 = self.nconv3_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_1(s_1, cs_1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv3_g(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        d_1, cd_1 = self.nconv3_d(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)

        # Stage 2
        s_2, cs_2 = self.npool_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_pool(s_1, cs_1)
        d_2, cd_2, gx_2, cgx_2, gy_2, cgy_2 = self.npool_dg(d_1, cd_1, s_1, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        s_2, cs_2 = self.nconv2_s(d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_1(s_2, cs_2)
        gx_2, cgx_2, gy_2, cgy_2 = self.nconv2_g(d_2, cd_2, s_2, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)
        d_2, cd_2 = self.nconv2_d(d_2, cd_2, s_2, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)
        s_2, cs_2 = self.nconv3_s(d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_1(s_2, cs_2)
        gx_2, cgx_2, gy_2, cgy_2 = self.nconv3_g(d_2, cd_2, s_2, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)
        d_2, cd_2 = self.nconv3_d(d_2, cd_2, s_2, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)

        # Stage 3
        s_3, cs_3 = self.npool_s(d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_pool(s_2, cs_2)
        d_3, cd_3, gx_3, cgx_3, gy_3, cgy_3 = self.npool_dg(d_2, cd_2, s_2, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)
        s_3, cs_3 = self.nconv2_s(d_3, cd_3, s_3, cs_3)
        s_prod = self.s_prod_1(s_3, cs_3)
        gx_3, cgx_3, gy_3, cgy_3 = self.nconv2_g(d_3, cd_3, s_3, cs_3, gx_3, cgx_3, gy_3, cgy_3, s_prod)
        d_3, cd_3 = self.nconv2_d(d_3, cd_3, s_3, cs_3, gx_3, cgx_3, gy_3, cgy_3, s_prod)

        # Stage 4
        s_4, cs_4 = self.npool_s(d_3, cd_3, s_3, cs_3)
        s_prod = self.s_prod_pool(s_3, cs_3)
        d_4, cd_4, gx_4, cgx_4, gy_4, cgy_4 = self.npool_dg(d_3, cd_3, s_3, cs_3, gx_3, cgx_3, gy_3, cgy_3, s_prod)
        s_4, cs_4 = self.nconv2_s(d_4, cd_4, s_4, cs_4)
        s_prod = self.s_prod_1(s_4, cs_4)
        gx_4, cgx_4, gy_4, cgy_4 = self.nconv2_g(d_4, cd_4, s_4, cs_4, gx_4, cgx_4, gy_4, cgy_4, s_prod)
        d_4, cd_4 = self.nconv2_d(d_4, cd_4, s_4, cs_4, gx_4, cgx_4, gy_4, cgy_4, s_prod)

        # Stage 3
        s_43, cs_43 = self.nup_s(s_4, cs_4)
        gx_43, cgx_43 = self.nup_gx(gx_4, cgx_4)
        gy_43, cgy_43 = self.nup_gy(gy_4, cgy_4)
        d_43, cd_43 = self.nup_d(d_4, cd_4)
        s_3, cs_3 = torch.cat((s_43, s_3), 1), torch.cat((cs_43, cs_3), 1)
        gx_3, cgx_3 = torch.cat((gx_43, gx_3), 1), torch.cat((cgx_43, cgx_3), 1)
        gy_3, cgy_3 = torch.cat((gy_43, gy_3), 1), torch.cat((cgy_43, cgy_3), 1)
        d_3, cd_3 = torch.cat((d_43, d_3), 1), torch.cat((cd_43, cd_3), 1)
        s_3, cs_3 = self.nconv4_s(d_3, cd_3, s_3, cs_3)
        s_prod = self.s_prod_2(s_3, cs_3).repeat(1, 2, 1, 1, 1)
        s_repeat = s_3.repeat(1, 2, 1, 1)
        gx_3, cgx_3, gy_3, cgy_3 = self.nconv4_g(d_3, cd_3, s_repeat, cs_3, gx_3, cgx_3, gy_3, cgy_3, s_prod)
        gx_repeat, cgx_repeat = gx_3.repeat(1, 2, 1, 1), cgx_3.repeat(1, 2, 1, 1)
        gy_repeat, cgy_repeat = gy_3.repeat(1, 2, 1, 1), cgy_3.repeat(1, 2, 1, 1)
        d_3, cd_3 = self.nconv4_d(d_3, cd_3, s_repeat, cs_3, gx_repeat, cgx_repeat, gy_repeat, cgy_repeat, s_prod)

        # Stage 2
        s_32, cs_32 = self.nup_s(s_3, cs_3)
        gx_32, cgx_32 = self.nup_gx(gx_3, cgx_3)
        gy_32, cgy_32 = self.nup_gy(gy_3, cgy_3)
        d_32, cd_32 = self.nup_d(d_3, cd_3)
        s_2, cs_2 = torch.cat((s_32, s_2), 1), torch.cat((cs_32, cs_2), 1)
        gx_2, cgx_2 = torch.cat((gx_32, gx_2), 1), torch.cat((cgx_32, cgx_2), 1)
        gy_2, cgy_2 = torch.cat((gy_32, gy_2), 1), torch.cat((cgy_32, cgy_2), 1)
        d_2, cd_2 = torch.cat((d_32, d_2), 1), torch.cat((cd_32, cd_2), 1)
        s_2, cs_2 = self.nconv5_s(d_2, cd_2, s_2, cs_2)
        s_prod = self.s_prod_2(s_2, cs_2).repeat(1, 2, 1, 1, 1)
        s_repeat = s_2.repeat(1, 2, 1, 1)
        gx_2, cgx_2, gy_2, cgy_2 = self.nconv5_g(d_2, cd_2, s_repeat, cs_2, gx_2, cgx_2, gy_2, cgy_2, s_prod)
        gx_repeat, cgx_repeat = gx_2.repeat(1, 2, 1, 1), cgx_2.repeat(1, 2, 1, 1)
        gy_repeat, cgy_repeat = gy_2.repeat(1, 2, 1, 1), cgy_2.repeat(1, 2, 1, 1)
        d_2, cd_2 = self.nconv5_d(d_2, cd_2, s_repeat, cs_2, gx_repeat, cgx_repeat, gy_repeat, cgy_repeat, s_prod)

        # Stage 1
        s_21, cs_21 = self.nup_s(s_2, cs_2)
        gx_21, cgx_21 = self.nup_gx(gx_2, cgx_2)
        gy_21, cgy_21 = self.nup_gy(gy_2, cgy_2)
        d_21, cd_21 = self.nup_d(d_2, cd_2)
        s_1, cs_1 = torch.cat((s_21, s_1), 1), torch.cat((cs_21, cs_1), 1)
        gx_1, cgx_1 = torch.cat((gx_21, gx_1), 1), torch.cat((cgx_21, cgx_1), 1)
        gy_1, cgy_1 = torch.cat((gy_21, gy_1), 1), torch.cat((cgy_21, cgy_1), 1)
        d_1, cd_1 = torch.cat((d_21, d_1), 1), torch.cat((cd_21, cd_1), 1)
        s_1, cs_1 = self.nconv6_s(d_1, cd_1, s_1, cs_1)
        s_prod = self.s_prod_2(s_1, cs_1).repeat(1, 2, 1, 1, 1)
        s_repeat = s_1.repeat(1, 2, 1, 1)
        gx_1, cgx_1, gy_1, cgy_1 = self.nconv6_g(d_1, cd_1, s_repeat, cs_1, gx_1, cgx_1, gy_1, cgy_1, s_prod)
        gx_repeat, cgx_repeat = gx_1.repeat(1, 2, 1, 1), cgx_1.repeat(1, 2, 1, 1)
        gy_repeat, cgy_repeat = gy_1.repeat(1, 2, 1, 1), cgy_1.repeat(1, 2, 1, 1)
        d_1, cd_1 = self.nconv6_d(d_1, cd_1, s_repeat, cs_1, gx_repeat, cgx_repeat, gy_repeat, cgy_repeat, s_prod)

        # Stage 0
        s_0, cs_0 = self.nup_s(s_1, cs_1)
        gx_0, cgx_0 = self.nup_gx(gx_1, cgx_1)
        gy_0, cgy_0 = self.nup_gy(gy_1, cgy_1)
        d_10, cd_10 = self.nup_d(d_1, cd_1)
        d_0, cd_0 = torch.cat((d_10, d_0), 1), torch.cat((cd_10, cd_0), 1)
        s_0, cs_0 = self.nconv6_s(d_0, cd_0, s_0, cs_0)
        s_prod = self.s_prod_2(s_0, cs_0).repeat(1, 2, 1, 1, 1)
        s_repeat = s_0.repeat(1, 2, 1, 1)
        gx_0, cgx_0, gy_0, cgy_0 = self.nconv6_g(d_0, cd_0, s_repeat, cs_0, gx_0, cgx_0, gy_0, cgy_0, s_prod)
        gx_repeat, cgx_repeat = gx_0.repeat(1, 2, 1, 1), cgx_0.repeat(1, 2, 1, 1)
        gy_repeat, cgy_repeat = gy_0.repeat(1, 2, 1, 1), cgy_0.repeat(1, 2, 1, 1)
        d_0, cd_0 = self.nconv6_d(d_0, cd_0, s_repeat, cs_0, gx_repeat, cgx_repeat, gy_repeat, cgy_repeat, s_prod)

        # output
        d, cd = self.nconv7_d(d_0, cd_0)
        return d, cd
