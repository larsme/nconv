
import torch
from modules.StructNConv.KernelChannels import KernelChannels


class s_prod_KernelChannels5(torch.nn.Module):
    def __init__(self):
        super(s_prod_KernelChannels5, self).__init__()
        self.kernel_channels = KernelChannels(5, 1, 2, 1)

    def forward(self, s, cs):
        '''
        :param
        s, cs to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        s of middle = 1, cs of middle = 1
        cs_out = product of cs along path to middle (including middle) with highest cs
        s_out = product of s along same path
        '''
        s_roll = self.kernel_channels.kernel_channels(s)
        cs_roll = self.kernel_channels.kernel_channels(cs)

        s11, c11 = s * s_roll[:, :, 1 * 5 + 1, :, :], cs * cs_roll[:, :, 1 * 5 + 1, :, :]
        s12, c12 = s * s_roll[:, :, 1 * 5 + 2, :, :], cs * cs_roll[:, :, 1 * 5 + 2, :, :]
        s13, c13 = s * s_roll[:, :, 1 * 5 + 3, :, :], cs * cs_roll[:, :, 1 * 5 + 3, :, :]
        s21, c21 = s * s_roll[:, :, 2 * 5 + 1, :, :], cs * cs_roll[:, :, 2 * 5 + 1, :, :]
        s23, c23 = s * s_roll[:, :, 2 * 5 + 3, :, :], cs * cs_roll[:, :, 2 * 5 + 3, :, :]
        s31, c31 = s * s_roll[:, :, 3 * 5 + 1, :, :], cs * cs_roll[:, :, 3 * 5 + 1, :, :]
        s32, c32 = s * s_roll[:, :, 3 * 5 + 2, :, :], cs * cs_roll[:, :, 3 * 5 + 2, :, :]
        s33, c33 = s * s_roll[:, :, 3 * 5 + 3, :, :], cs * cs_roll[:, :, 3 * 5 + 3, :, :]
        
        s00 = s11 * s_roll[:, :, 0 * 5 + 0, :, :]
        s04 = s13 * s_roll[:, :, 0 * 5 + 4, :, :]
        s40 = s31 * s_roll[:, :, 4 * 5 + 0, :, :]
        s44 = s33 * s_roll[:, :, 4 * 5 + 4, :, :]
        
        s01 = torch.stack((s11, s12), dim=2).gather(dim=2,index=torch.argmax(torch.stack((c11, c12), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 0 * 5 + 1, :, :]
        s02 = torch.stack((s11, s12, s13), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c11, c12, c13), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 0 * 5 + 2, :, :]
        s03 = torch.stack((s12, s13), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c12, c13), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 0 * 5 + 3, :, :]
        
        s10 = torch.stack((s11, s21), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c11, c21), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 1 * 5 + 0, :, :]
        s20 = torch.stack((s11, s21, s31), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c11, c21, c31), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 2 * 5 + 0, :, :]
        s30 = torch.stack((s21, s31), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c21, c31), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 3 * 5 + 0, :, :]

        s14 = torch.stack((s13, s23), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c13, c23), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 1 * 5 + 4, :, :]
        s24 = torch.stack((s13, s23, s33), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c13, c23, c33), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 2 * 5 + 4, :, :]
        s34 = torch.stack((s23, s33), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c23, c33), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 3 * 5 + 4, :, :]

        s41 = torch.stack((s31, s32), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c31, c32), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 4 * 5 + 1, :, :]
        s42 = torch.stack((s31, s32, s33), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c31, c32, c33), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 4 * 5 + 2, :, :]
        s43 = torch.stack((s32, s33), dim = 2).gather(dim=2,index=torch.argmax(torch.stack((c32, c33), dim = 2), dim=2, keepdim=True)).squeeze(2) * s_roll[:, :, 4 * 5 + 3, :, :]


        return torch.stack((s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, torch.ones_like(s00), s23, s24, s30, s31, s32, s33, s34, s40, s41, s42, s43, s44), dim = 2)
