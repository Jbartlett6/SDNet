import torch.nn as nn
import torch

class SmallSHConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self, activation):
    

        
        self.casc = nn.Sequential(nn.Conv3d(47, 64, 3, padding='same'),  
                                nn.BatchNorm3d(64),
                                activation,  
                                nn.Conv3d(64, 128, 3, padding='same'),
                                nn.BatchNorm3d(128),
                                activation,
                                nn.Conv3d(128, 256, 3, padding='same'),
                                nn.BatchNorm3d(256),  
                                activation,
                                nn.Conv3d(256, 512, 3),
                                activation,
                                nn.Conv3d(512, 94, 1, padding = 'same'))


    def forward(self, x):
        return self.casc(x)

class GLUConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self):
        super().__init__()
        
        self.casc = nn.Sequential(nn.Conv3d(47, 128, 3, padding='same'),  
                                nn.BatchNorm3d(128),
                                torch.nn.GLU(dim=1),  
                                nn.Conv3d(64, 2*128, 3, padding='same'),
                                nn.BatchNorm3d(2*128),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(128, 2*192, 3, padding='same'),
                                nn.BatchNorm3d(2*192),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(192, 2*256, 3, padding='same'),
                                nn.BatchNorm3d(2*256),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(256, 2*320, 3, padding='same'),
                                nn.BatchNorm3d(2*320),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(320, 2*384, 3, padding='same'),
                                nn.BatchNorm3d(2*384),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(384, 2*448, 3, padding='same'),
                                nn.BatchNorm3d(2*448),
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(448, 2*512, 3),   
                                torch.nn.GLU(dim=1),
                                nn.Conv3d(512, 94, 1, padding = 'same'))
        
        

    
    def forward(self, x):
        x = x.transpose(1,4).squeeze()
        x = self.casc(x)
        x = x.transpose(1,4).unsqueeze(5)
        return x

class SkipConnectConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self):
        super().__init__()
        
        self.casc = nn.Sequential(nn.Conv3d(47, 64, 3, padding='same'),  
                                nn.BatchNorm3d(64),
                                nn.ReLU(inplace=True),  
                                nn.Conv3d(64, 128, 3, padding='same'),
                                nn.BatchNorm3d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(128, 192, 3, padding='same'),
                                nn.BatchNorm3d(192),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(192, 256, 3, padding='same'),
                                nn.BatchNorm3d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(256, 320, 3, padding='same'),
                                nn.BatchNorm3d(320),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(320, 384, 3, padding='same'),
                                nn.BatchNorm3d(384),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(384, 448, 3, padding='same'),
                                nn.BatchNorm3d(448),
                                nn.ReLU(inplace=True))
                                
                                
        self.feature_conv_1 = nn.Conv3d(448+512, 512, 3)   
                                #nn.ReLU(inplace=True),
        self.feature_conv_2 = nn.Conv3d(512, 94, 1, padding = 'same')
        
        

    
    def forward(self, x, prev_feat):
        x = self.casc(x)
        x = torch.cat((x,prev_feat), dim = 1)
        x = self.feature_conv_1(x)
        curr_feat = torch.relu(x)
        x = self.feature_conv_2(x)
        return x, curr_feat


class SHConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self, activation):
        super().__init__()
        
        self.casc = nn.Sequential(nn.Conv3d(47, 64, 3, padding='same'),  
                                nn.BatchNorm3d(64),
                                activation,  
                                nn.Conv3d(64, 128, 3, padding='same'),
                                nn.BatchNorm3d(128),
                                activation,
                                nn.Conv3d(128, 192, 3, padding='same'),
                                nn.BatchNorm3d(192),
                                activation,
                                nn.Conv3d(192, 256, 3, padding='same'),
                                nn.BatchNorm3d(256),
                                activation,
                                nn.Conv3d(256, 320, 3, padding='same'),
                                nn.BatchNorm3d(320),
                                activation,
                                nn.Conv3d(320, 384, 3, padding='same'),
                                nn.BatchNorm3d(384),
                                activation,
                                nn.Conv3d(384, 448, 3, padding='same'),
                                nn.BatchNorm3d(448),
                                activation,
                                nn.Conv3d(448, 512, 3),   
                                activation,
                                nn.Conv3d(512, 94, 1, padding = 'same'))
        
        

    
    def forward(self, x):
        x = x.transpose(1,4).squeeze()
        x = self.casc(x)
        x = x.transpose(1,4).unsqueeze(5)
        return x