########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch
import torch.nn.functional as F

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
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        # pool
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_e_with_d(in_channels=2 * num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1)])
        # pooling
        if maxpool_e:
            self.npool_e = StructNMaxPool2D_e(channels=num_channels,
                                              kernel_size=3, stride=2, padding=1,
                                              devalue_pooled_confidence=devalue_pooled_confidence)
        else:
            self.npool_e = StructNConv2D_e_with_d(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1, dilation=1,
                                                  devalue_pooled_confidence=devalue_pooled_confidence)
        if nn_upsample_e:
            self.nup_e = NearestNeighbourUpsample2(kernel_size=3, stride=2, padding=1)
        else:
            self.nup_e = StructNDeconv2D_e(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1, dilation=1)


        # e_prod
        self.e_prod_1 = e_prod_KernelChannels5()
        if maxpool_d:
            self.e_prod_pool = ReturnNone()
        else:
            self.e_prod_pool = e_prod_KernelChannels3(stride=2)
        self.e_prod_2 = e_prod_KernelChannels3()


        # depth modules
        # in_channels not 1 because of multiplication with output of nconv1_e
        self.nconv_d = torch.nn.ModuleList([StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=5, stride=1, padding=2, dilation=1),
                        StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
                        StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                               kernel_size=3, stride=1, padding=1, dilation=1),
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
            self.npool_d = StructNMaxPool2D_d_with_e(kernel_size=3, stride=2, padding=1)
        else:
            self.npool_d = StructNConv2D_d_with_s(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1, dilation=1)
        if nn_upsample_d:
            self.nup_d = NearestNeighbourUpsample(kernel_size=3, stride=2, padding=1)
        else:
            self.nup_d = StructNDeconv2D(in_channels=num_channels, out_channels=num_channels, init_method=params['init_method'], mirror_weights=params['mirror_weights'],
                                                  kernel_size=3, stride=2, padding=1, dilation=1)
        self.outs = ['d', 'cd', 'e', 'ce']
    
    def print(self):
        s = 'nconv_d\n'
        s+= 'spatial\n'
        s_list = []
        for i in range(5):
            s_list.append('')
        for nconv_d in self.nconv_d:
            s_list2 = []
            for i in range(5):
                s_list2.append('')
            s_list2= nconv_d.print(s_list2)
            for i in range(5):
                s_list[i] += '{:<30}'.format(s_list2[i]) + '| '
        for i in range(5):
            s+= s_list[i]+'\n'
        s+='\n'

        s+='nconv_e\n'
        title_rows = set()
        s_list = []
        for i in range(4):
            title_rows.add(len(s_list))
            s_list.append('spatial {0}'.format(['/','-','\\','|'][i]))
            for j in range(5):
                s_list.append('') 
                
        title_rows.add(len(s_list))
        s_list.append('') 
        title_rows.add(len(s_list))
        s_list.append('channel_weights prev / - \\ | to  / - \\ |')
        for dir in range(4):
            s_list.append('')
        title_rows.add(len(s_list))
        s_list.append('channel_weights skip / - \\ | to  / - \\ |')
        for dir in range(4):
            s_list.append('')
            
        title_rows.add(len(s_list))
        s_list.append('')
        title_rows.add(len(s_list))
        s_list.append('pow for prev / - \\ | per side')
        for dir in range(4):
            s_list.append('')
        title_rows.add(len(s_list))
        s_list.append('pow for skip / - \\ | per side')
        for dir in range(4):
            s_list.append('')
            
        title_rows.add(len(s_list))
        s_list.append('')            
        title_rows.add(len(s_list))
        s_list.append('w prop (prev, skip)')
        for dir in range(2):
            s_list.append('')

        for nconv_e in self.nconv_e:
            s_list2 = []
            for i in range(len(s_list)):
                s_list2.append('')
            s_list2= nconv_e.print(s_list2)
            for i in range(len(s_list2)):
                if i not in title_rows:
                    s_list[i] += '{:<30}'.format(s_list2[i]) + '| '
        for i in range(len(s_list)):
            s+= s_list[i]+'\n'
        print(s)

        
    def prep_eval(self):
        for nconv in self.nconv_d:
            nconv.prep_eval()
        for nconv in self.nconv_e:
            nconv.prep_eval()
        self.npool_d.prep_eval()
        self.nup_d.prep_eval()
        self.npool_e.prep_eval()
        self.nup_e.prep_eval()
    
    def forward(self, d_0, cd_0=None, e_0=None, ce_0=None):
        

        if cd_0 is None:
            cd_0 = (d_0 > 0).float()
        if e_0 is None:
            e_0 = ce_0 = torch.zeros(size=(d_0.shape[0], d_0.shape[1], 4, d_0.shape[2], d_0.shape[3]), device=d_0.device)  

        # Stage 0
        e_0, ce_0 = self.nconv_e[0](d_0, cd_0, e_0, ce_0)
        e_prod = self.e_prod_1(e_0, ce_0)
        d_0, cd_0 = self.nconv_d[0](d_0, cd_0, e_0, ce_0, e_prod)

        # Stage 1
        e_1, ce_1 = self.npool_e(d_0, cd_0, e_0, ce_0)
        e_prod = self.e_prod_pool(e_0, ce_0)
        d_1, cd_1 = self.npool_d(d_0, cd_0, e_0, ce_0, e_prod)
        e_1, ce_1 = self.nconv_e[1](d_1, cd_1, e_1, ce_1)
        e_prod = self.e_prod_2(e_1, ce_1)
        d_1, cd_1 = self.nconv_d[1](d_1, cd_1,  e_1, ce_1, e_prod)

        # Stage 2
        e_2, ce_2 = self.npool_e(d_1, cd_1, e_1, ce_1)
        e_prod = self.e_prod_pool(e_1, ce_1)
        d_2, cd_2 = self.npool_d(d_1, cd_1, e_1, ce_1, e_prod)
        e_2, ce_2 = self.nconv_e[2](d_2, cd_2, e_2, ce_2)
        e_prod = self.e_prod_2(e_2, ce_2)
        d_2, cd_2 = self.nconv_d[2](d_2, cd_2,  e_2, ce_2, e_prod)

        # Stage 3
        e_3, ce_3 = self.npool_e(d_2, cd_2, e_2, ce_2)
        e_prod = self.e_prod_pool(e_2, ce_2)
        d_3, cd_3 = self.npool_d(d_2, cd_2, e_2, ce_2, e_prod)
        e_3, ce_3 = self.nconv_e[3](d_3, cd_3, e_3, ce_3)
        e_prod = self.e_prod_2(e_3, ce_3)
        d_3, cd_3 = self.nconv_d[3](d_3, cd_3, e_3, ce_3, e_prod)

        # Stage 2
        e_32, ce_32 = self.nup_e(e_3, ce_3, e_2.shape)
        d_32, cd_32 = self.nup_d(d_3, cd_3, d_2.shape)
        e_2, ce_2 = torch.cat((e_32, e_2), 1), torch.cat((ce_32, ce_2), 1)
        d_2, cd_2 = torch.cat((d_32, d_2), 1), torch.cat((cd_32, cd_2), 1)
        e_2, ce_2 = self.nconv_e[4](d_2, cd_2, e_2, ce_2)
        e_prod = self.e_prod_2(e_2, ce_2)
        d_2, cd_2 = self.nconv_d[4](d_2, cd_2, e_2, ce_2, e_prod)

        # Stage 1
        e_21, ce_21 = self.nup_e(e_2, ce_2, e_1.shape)
        d_21, cd_21 = self.nup_d(d_2, cd_2, d_1.shape)
        e_1, ce_1 = torch.cat((e_21, e_1), 1), torch.cat((ce_21, ce_1), 1)
        d_1, cd_1 = torch.cat((d_21, d_1), 1), torch.cat((cd_21, cd_1), 1)
        e_1, ce_1 = self.nconv_e[5](d_1, cd_1, e_1, ce_1)
        e_prod = self.e_prod_2(e_1, ce_1)
        d_1, cd_1 = self.nconv_d[5](d_1, cd_1, e_1, ce_1, e_prod)

        # Stage 0
        e_10, ce_10 = self.nup_e(e_1, ce_1, e_0.shape)
        d_10, cd_10 = self.nup_d(d_1, cd_1, d_0.shape)
        e_0, ce_0 = torch.cat((e_10, e_0), 1), torch.cat((ce_10, ce_0), 1)
        d_0, cd_0 = torch.cat((d_10, d_0), 1), torch.cat((cd_10, cd_0), 1)
        e_0, ce_0 = self.nconv_e[6](d_0, cd_0, e_0, ce_0)
        e_prod = self.e_prod_2(e_0, ce_0)
        d_0, cd_0 = self.nconv_d[6](d_0, cd_0, e_0, ce_0, e_prod)

        # output
        d, cd = self.nconv_d[7](d_0, cd_0)
        s, cs = self.nconv_d[7](e_0.view(d.shape[0], -1, d.shape[2] * 4, d.shape[3]), ce_0.view(d.shape[0], -1, d.shape[2] * 4, d.shape[3]))
        s[cs == 0] = 0
        return d, cd, s.view(d.shape[0], 4, d.shape[2], d.shape[3]), cs.view(d.shape[0], 4, d.shape[2], d.shape[3])
