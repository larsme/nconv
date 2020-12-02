########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
import torch

from workspace.StructNConv.Unguided_de.Unguided_de import CNN as Unguided_de


class CNN(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        self.net1 = Unguided_de(params)
        checkpoint_dict = torch.load(params['net1_checkpoint'])
        self.net1.load_state_dict(checkpoint_dict['net'])        
        # Disable Training for net 1
        for p in self.net1.parameters():            
            p.requires_grad=False
        self.net1.training=False

        self.net2 = Unguided_de(params)
        
        self.outs = ['d', 'cd', 'e', 'ce']

    def enforce_limits(self):
        self.net2.enforce_limits()
    
    
    def forward(self, d_0, cd_0, e_0=None, ce_0 = None):
        _, _, e_0, ce_0 = self.net1(d_0, cd_0, e_0, ce_0)
        return self.net2(d_0, cd_0, e_0.unsqueeze(1), ce_0.unsqueeze(1))
