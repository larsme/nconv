
import torch
import torch.nn.functional as F
from modules.StructNConv.e_prod_KernelChannels3 import e_prod_KernelChannels3


class e_prod_KernelChannels5(torch.nn.Module):
    def __init__(self):
        super(e_prod_KernelChannels5, self).__init__()
        self.e_prod_KernelChannels3 = e_prod_KernelChannels3()
        self.eps = 1e-20

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

        s3, c3 = self.e_prod_KernelChannels3(s, cs), self.e_prod_KernelChannels3(cs, s)
        s3_pad, c3_pad = F.pad(s3, (1,1,1,1)), F.pad(c3, (1,1,1,1))
        
        # edges
        s00 = s3[:,:,0,:,:].clone()* s3_pad[:,:,0,:-2,:-2]
        s04 = s3[:,:,2,:,:].clone()* s3_pad[:,:,2,:-2,2:]
        s40 = s3[:,:,6,:,:].clone()* s3_pad[:,:,6,2:,:-2]
        s44 = s3[:,:,8,:,:].clone()* s3_pad[:,:,8,2:,2:]
        
        s10 = torch.stack((s3[:,:,0,:,:].clone()*s3_pad[:,:,3,:-2,:-2], s3[:,:,3,:,:].clone()*s3_pad[:,:,0,1:-1,:-2]), dim=2)
        cs10= torch.stack((c3[:,:,0,:,:].clone()*c3_pad[:,:,3,:-2,:-2], c3[:,:,3,:,:].clone()*c3_pad[:,:,0,1:-1,:-2]), dim=2)
        s20 = torch.stack((s3[:,:,0,:,:].clone()*s3_pad[:,:,6,:-2,:-2], s3[:,:,3,:,:].clone()*s3_pad[:,:,3,1:-1,:-2], s3[:,:,6,:,:].clone()*s3_pad[:,:,0,2:,:-2]), dim=2)
        cs20= torch.stack((c3[:,:,0,:,:].clone()*c3_pad[:,:,6,:-2,:-2], c3[:,:,3,:,:].clone()*c3_pad[:,:,3,1:-1,:-2], c3[:,:,6,:,:].clone()*c3_pad[:,:,0,2:,:-2]), dim=2)
        s30 = torch.stack((                                             s3[:,:,3,:,:].clone()*s3_pad[:,:,6,1:-1,:-2], s3[:,:,6,:,:].clone()*s3_pad[:,:,3,2:,:-2]), dim=2)
        cs30= torch.stack((                                             c3[:,:,3,:,:].clone()*c3_pad[:,:,6,1:-1,:-2], c3[:,:,6,:,:].clone()*c3_pad[:,:,3,2:,:-2]), dim=2)

        s14 = torch.stack((s3[:,:,2,:,:].clone()*s3_pad[:,:,5,:-2,2:], s3[:,:,5,:,:].clone()*s3_pad[:,:,2,1:-1,2:]), dim=2)
        cs14= torch.stack((c3[:,:,2,:,:].clone()*c3_pad[:,:,5,:-2,2:], c3[:,:,5,:,:].clone()*c3_pad[:,:,2,1:-1,2:]), dim=2)
        s24 = torch.stack((s3[:,:,2,:,:].clone()*s3_pad[:,:,8,:-2,2:], s3[:,:,5,:,:].clone()*s3_pad[:,:,5,1:-1,2:], s3[:,:,8,:,:].clone()*s3_pad[:,:,2,2:,2:]), dim=2)
        cs24= torch.stack((c3[:,:,2,:,:].clone()*c3_pad[:,:,8,:-2,2:], c3[:,:,5,:,:].clone()*c3_pad[:,:,3,1:-1,2:], c3[:,:,8,:,:].clone()*c3_pad[:,:,2,2:,2:]), dim=2)
        s34 = torch.stack((                                            s3[:,:,5,:,:].clone()*s3_pad[:,:,8,1:-1,2:], s3[:,:,8,:,:].clone()*s3_pad[:,:,5,2:,2:]), dim=2)
        cs34= torch.stack((                                            c3[:,:,5,:,:].clone()*c3_pad[:,:,8,1:-1,2:], c3[:,:,8,:,:].clone()*c3_pad[:,:,5,2:,2:]), dim=2)
        
        s01 = torch.stack((s3[:,:,0,:,:].clone()*s3_pad[:,:,1,:-2,:-2], s3[:,:,1,:,:].clone()*s3_pad[:,:,0,:-2,1:-1]), dim=2)
        cs01= torch.stack((c3[:,:,0,:,:].clone()*c3_pad[:,:,1,:-2,:-2], c3[:,:,1,:,:].clone()*c3_pad[:,:,0,:-2,1:-1]), dim=2)
        s02 = torch.stack((s3[:,:,0,:,:].clone()*s3_pad[:,:,2,:-2,:-2], s3[:,:,1,:,:].clone()*s3_pad[:,:,1,:-2,1:-1], s3[:,:,2,:,:].clone()*s3_pad[:,:,0,:-2,2:]), dim=2)
        cs02= torch.stack((c3[:,:,0,:,:].clone()*c3_pad[:,:,2,:-2,:-2], c3[:,:,1,:,:].clone()*c3_pad[:,:,1,:-2,1:-1], c3[:,:,2,:,:].clone()*c3_pad[:,:,0,:-2,2:]), dim=2)
        s03 = torch.stack((                                             s3[:,:,1,:,:].clone()*s3_pad[:,:,2,:-2,1:-1], s3[:,:,2,:,:].clone()*s3_pad[:,:,1,:-2,2:]), dim=2)
        cs03= torch.stack((                                             c3[:,:,1,:,:].clone()*c3_pad[:,:,2,:-2,1:-1], c3[:,:,2,:,:].clone()*c3_pad[:,:,1,:-2,2:]), dim=2)

        s41 = torch.stack((s3[:,:,6,:,:].clone()*s3_pad[:,:,7,2:,:-2], s3[:,:,7,:,:].clone()*s3_pad[:,:,6,2:,1:-1]), dim=2)
        cs41= torch.stack((c3[:,:,6,:,:].clone()*c3_pad[:,:,7,2:,:-2], c3[:,:,7,:,:].clone()*c3_pad[:,:,6,2:,1:-1]), dim=2)
        s42 = torch.stack((s3[:,:,6,:,:].clone()*s3_pad[:,:,8,2:,:-2], s3[:,:,7,:,:].clone()*s3_pad[:,:,7,2:,1:-1], s3[:,:,8,:,:].clone()*s3_pad[:,:,6,2:,2:]), dim=2)
        cs42= torch.stack((c3[:,:,6,:,:].clone()*c3_pad[:,:,8,2:,:-2], c3[:,:,7,:,:].clone()*c3_pad[:,:,7,2:,1:-1], c3[:,:,8,:,:].clone()*c3_pad[:,:,6,2:,2:]), dim=2)
        s43 = torch.stack((                                            s3[:,:,7,:,:].clone()*s3_pad[:,:,8,2:,1:-1], s3[:,:,8,:,:].clone()*s3_pad[:,:,7,2:,2:]), dim=2)
        cs43= torch.stack((                                            c3[:,:,7,:,:].clone()*c3_pad[:,:,8,2:,1:-1], c3[:,:,8,:,:].clone()*c3_pad[:,:,7,2:,2:]), dim=2)


        return torch.stack((s00,                                        (s01*cs01).sum(2)/(cs02.sum(2) + self.eps), (s02*cs02).sum(2)/(cs02.sum(2) + self.eps), (s03*cs03).sum(2)/(cs03.sum(2) + self.eps), s04, 
                            (s10*cs10).sum(2)/(cs10.sum(2) + self.eps), s3[:,:,0,:,:],                              s3[:,:,1,:,:],                              s3[:,:,2,:,:],                              (s14*cs14).sum(2)/(cs14.sum(2) + self.eps), 
                            (s20*cs20).sum(2)/(cs20.sum(2) + self.eps), s3[:,:,3,:,:],                              s3[:,:,4,:,:],                              s3[:,:,5,:,:],                              (s24*cs24).sum(2)/(cs24.sum(2) + self.eps),
                            (s30*cs30).sum(2)/(cs20.sum(2) + self.eps), s3[:,:,6,:,:],                              s3[:,:,7,:,:],                              s3[:,:,8,:,:],                              (s34*cs34).sum(2)/(cs34.sum(2) + self.eps),
                            s40,                                        (s41*cs41).sum(2)/(cs41.sum(2) + self.eps), (s42*cs42).sum(2)/(cs42.sum(2) + self.eps), (s43*cs43).sum(2)/(cs43.sum(2) + self.eps), s44), dim = 2)
