import torch.nn as nn

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

class SHConvCascadeLayer_MS(nn.Module):
    """Cascade Layer"""
    def __init__(self, activation):
        super().__init__()
        
        self.casc = nn.Sequential(nn.Conv3d(94, 128, 3, padding='same'),
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

