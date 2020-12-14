
import torch
import torch.nn.functional as F
from modules.StructNConv.e_prod_KernelChannels3 import e_prod_KernelChannels3


class e_prod_KernelChannels5(torch.nn.Module):
    def __init__(self):
        super(e_prod_KernelChannels5, self).__init__()
        self.e_prod_KernelChannels3 = e_prod_KernelChannels3()

    def forward(self, e, ce):
        '''
        :param
        e, ce to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        e of middle = 1, ce of middle = 1
        cs_out = product of ce along path to middle (including middle) with highest ce
        s_out = product of e along same path
        '''

        e3, ce3 = self.e_prod_KernelChannels3(e, ce), self.e_prod_KernelChannels3(ce, e)
        e3_pad, ce3_pad = F.pad(e3, (1,1,1,1)), F.pad(ce3, (1,1,1,1))
        
        # edges
        e00 = e3[:,:,0,:,:].clone()* e3_pad[:,:,0,:-2,:-2]
        e04 = e3[:,:,2,:,:].clone()* e3_pad[:,:,2,:-2,2:]
        e40 = e3[:,:,6,:,:].clone()* e3_pad[:,:,6,2:,:-2]
        e44 = e3[:,:,8,:,:].clone()* e3_pad[:,:,8,2:,2:]
        
        e10 = torch.stack((e3[:,:,0,:,:].clone()*e3_pad[:,:,3,:-2,:-2],   e3[:,:,3,:,:].clone()*e3_pad[:,:,0,1:-1,:-2]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,0,:,:].clone()*ce3_pad[:,:,3,:-2,:-2], ce3[:,:,3,:,:].clone()*ce3_pad[:,:,0,1:-1,:-2]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e20 = torch.stack((e3[:,:,0,:,:].clone()*e3_pad[:,:,6,:-2,:-2],   e3[:,:,3,:,:].clone()*e3_pad[:,:,3,1:-1,:-2],   e3[:,:,6,:,:].clone()*e3_pad[:,:,0,2:,:-2]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,0,:,:].clone()*ce3_pad[:,:,6,:-2,:-2], ce3[:,:,3,:,:].clone()*ce3_pad[:,:,3,1:-1,:-2], ce3[:,:,6,:,:].clone()*ce3_pad[:,:,0,2:,:-2]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e30 = torch.stack((                                               e3[:,:,3,:,:].clone()*e3_pad[:,:,6,1:-1,:-2],   e3[:,:,6,:,:].clone()*e3_pad[:,:,3,2:,:-2]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((                                               ce3[:,:,3,:,:].clone()*ce3_pad[:,:,6,1:-1,:-2], ce3[:,:,6,:,:].clone()*ce3_pad[:,:,3,2:,:-2]), dim=2), dim=2, keepdim=True)).squeeze(2)

        e14 = torch.stack((e3[:,:,2,:,:].clone()*e3_pad[:,:,5,:-2,2:],   e3[:,:,5,:,:].clone()*e3_pad[:,:,2,1:-1,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,2,:,:].clone()*ce3_pad[:,:,5,:-2,2:], ce3[:,:,5,:,:].clone()*ce3_pad[:,:,2,1:-1,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e24 = torch.stack((e3[:,:,2,:,:].clone()*e3_pad[:,:,8,:-2,2:],   e3[:,:,5,:,:].clone()*e3_pad[:,:,5,1:-1,2:],   e3[:,:,8,:,:].clone()*e3_pad[:,:,2,2:,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,2,:,:].clone()*ce3_pad[:,:,8,:-2,2:], ce3[:,:,5,:,:].clone()*ce3_pad[:,:,3,1:-1,2:], ce3[:,:,8,:,:].clone()*ce3_pad[:,:,2,2:,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e34 = torch.stack((                                              e3[:,:,5,:,:].clone()*e3_pad[:,:,8,1:-1,2:],   e3[:,:,8,:,:].clone()*e3_pad[:,:,5,2:,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((                                              ce3[:,:,5,:,:].clone()*ce3_pad[:,:,8,1:-1,2:], ce3[:,:,8,:,:].clone()*ce3_pad[:,:,5,2:,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)
        
        e01 = torch.stack((e3[:,:,0,:,:].clone()*e3_pad[:,:,1,:-2,:-2],   e3[:,:,1,:,:].clone()*e3_pad[:,:,0,:-2,1:-1]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,0,:,:].clone()*ce3_pad[:,:,1,:-2,:-2], ce3[:,:,1,:,:].clone()*ce3_pad[:,:,0,:-2,1:-1]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e02 = torch.stack((e3[:,:,0,:,:].clone()*e3_pad[:,:,2,:-2,:-2],   e3[:,:,1,:,:].clone()*e3_pad[:,:,1,:-2,1:-1],   e3[:,:,2,:,:].clone()*e3_pad[:,:,0,:-2,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,0,:,:].clone()*ce3_pad[:,:,2,:-2,:-2], ce3[:,:,1,:,:].clone()*ce3_pad[:,:,1,:-2,1:-1], ce3[:,:,2,:,:].clone()*ce3_pad[:,:,0,:-2,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e03 = torch.stack((                                               e3[:,:,1,:,:].clone()*e3_pad[:,:,2,:-2,1:-1],   e3[:,:,2,:,:].clone()*e3_pad[:,:,1,:-2,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((                                               ce3[:,:,1,:,:].clone()*ce3_pad[:,:,2,:-2,1:-1], ce3[:,:,2,:,:].clone()*ce3_pad[:,:,1,:-2,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)

        e41 = torch.stack((e3[:,:,6,:,:].clone()*e3_pad[:,:,7,2:,:-2],   e3[:,:,7,:,:].clone()*e3_pad[:,:,6,2:,1:-1]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,6,:,:].clone()*ce3_pad[:,:,7,2:,:-2], ce3[:,:,7,:,:].clone()*ce3_pad[:,:,6,2:,1:-1]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e42 = torch.stack((e3[:,:,6,:,:].clone()*e3_pad[:,:,8,2:,:-2],   e3[:,:,7,:,:].clone()*e3_pad[:,:,7,2:,1:-1],   e3[:,:,8,:,:].clone()*e3_pad[:,:,6,2:,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((ce3[:,:,6,:,:].clone()*ce3_pad[:,:,8,2:,:-2], ce3[:,:,7,:,:].clone()*ce3_pad[:,:,7,2:,1:-1], ce3[:,:,8,:,:].clone()*ce3_pad[:,:,6,2:,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)
        e43 = torch.stack((                                              e3[:,:,7,:,:].clone()*e3_pad[:,:,8,2:,1:-1],   e3[:,:,8,:,:].clone()*e3_pad[:,:,7,2:,2:]), dim=2).gather(dim=2, index=torch.argmax(
              torch.stack((                                              ce3[:,:,7,:,:].clone()*ce3_pad[:,:,8,2:,1:-1], ce3[:,:,8,:,:].clone()*ce3_pad[:,:,7,2:,2:]), dim=2), dim=2, keepdim=True)).squeeze(2)


        return torch.stack((e00, e01,           e02,           e03,           e04, 
                            e10, e3[:,:,0,:,:], e3[:,:,1,:,:], e3[:,:,2,:,:], e14, 
                            e20, e3[:,:,3,:,:], e3[:,:,4,:,:], e3[:,:,5,:,:], e24,
                            e30, e3[:,:,6,:,:], e3[:,:,7,:,:], e3[:,:,8,:,:], e34,
                            e40, e41,           e42,           e43,           e44), dim = 2)
