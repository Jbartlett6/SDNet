import torch.nn as nn
import torch


# class FixelNet(nn.Module):
#     """Cascade Layer"""

#     def __init__(self):
#         super().__init__()
#         self.casc = nn.Sequential(nn.Linear(47, 512),
#                                   nn.BatchNorm1d(512),  
#                                   nn.ReLU(inplace=True),  
#                                   nn.Linear(512,256),
#                                   nn.BatchNorm1d(256),
#                                   nn.ReLU(inplace=True),
#                                   nn.Linear(256,128),
#                                   nn.BatchNorm1d(128),  
#                                   nn.ReLU(inplace=True),
#                                   nn.Linear(128,64),
#                                   nn.ReLU(inplace=True),
#                                   nn.Linear(64,13))

#     def forward(self, x):
#         return self.casc(x)



class FixelNet(nn.Module):
    """Cascade Layer"""

    def __init__(self):
        super().__init__()
        self.casc = nn.Sequential(nn.Linear(45, 1000),
                                  nn.BatchNorm1d(1000),  
                                  nn.ReLU(inplace=True),  
                                  nn.Linear(1000,800),
                                  nn.BatchNorm1d(800),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(800,600),
                                  nn.BatchNorm1d(600),  
                                  nn.ReLU(inplace=True),
                                  nn.Linear(600,400),
                                  nn.BatchNorm1d(400),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(400,200),
                                  nn.BatchNorm1d(200),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(200,100),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(100,5))

    def forward(self, x):
        return self.casc(x)

def init_fixnet(opts):
    #Initialising the network and loading the parameters:
    parameter_path = '/bask/projects/d/duanj-ai-imaging/jxb1336/code/fixel_loss/checkpoints/sh-bignet/model_dict.pt'
    net = FixelNet()
    net.load_state_dict(torch.load(parameter_path))
    
    #Setting the network to be used appropriately as loss
    net.eval()
    net.requires_grad_(False)
    net.to(opts.device)


    return net