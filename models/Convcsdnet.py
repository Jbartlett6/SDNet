from socket import AF_ATMPVC
import sys
import os 
sys.path.append(os.path.join(sys.path[0],'..'))
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.special
import nibabel as nib
import random 
import matplotlib.pyplot as plt 
import time
import numpy as np
import util 
import data

class ConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self, dc):
        super().__init__()
        
        if dc=='CSD':
            
            self.casc = nn.Sequential(nn.Conv3d(47, 80, 3, padding='same'),  
                                    nn.BatchNorm3d(80),
                                    nn.ReLU(inplace=True),  
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.BatchNorm3d(80),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.BatchNorm3d(80),  
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 47, 3))
        else:
            self.casc = nn.Sequential(nn.Conv3d(47, 80, 3, padding='same'), 
                                    nn.BatchNorm3d(80), 
                                    nn.ReLU(inplace=True),  
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.BatchNorm3d(80),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.BatchNorm3d(80),  
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 80, 3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(80, 300, 3))

    def forward(self, x):
        return self.casc(x)
    
class SHNormCascadeLayer(nn.module):

class LargeConvCascadeLayer(nn.Module):
    """Cascade Layer"""
    def __init__(self, dc):
        super().__init__()
        if dc=='CSD':
            self.casc = nn.Sequential(nn.Conv3d(47, 256, 3, padding='same'),  
                                    nn.BatchNorm3d(256),
                                    nn.ReLU(inplace=True),  
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.BatchNorm3d(256),  
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 47, 3))
        else:
            self.casc = nn.Sequential(nn.Conv3d(47, 256, 3, padding='same'), 
                                    nn.BatchNorm3d(256), 
                                    nn.ReLU(inplace=True),  
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.BatchNorm3d(256),  
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 256, 3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(256, 300, 3))

    def forward(self, x):
        return self.casc(x)


class FCNet(nn.Module):
    def __init__(self, opts):
        super(FCNet, self).__init__()

        self.opts = opts

        if opts.learn_lambda == True:
            self.deep_reg = nn.Parameter(torch.tensor(self.opts.deep_reg))
            self.neg_reg = nn.Parameter(torch.tensor(self.opts.neg_reg))
            self.alpha = nn.Parameter(torch.tensor(self.opts.alpha).float())
        else:
            #Data consistency term parameters:
            self.register_buffer('deep_reg', torch.tensor(self.opts.deep_reg))
            self.register_buffer('neg_reg', torch.tensor(self.opts.neg_reg))
            self.register_buffer('alpha', torch.tensor(self.opts.alpha).float())


        self.sampling_directions = torch.load(os.path.join('utils/300_predefined_directions.pt'))
        
        self.order = 8 
        P = util.construct_sh_basis(self.sampling_directions, self.order)
        P_temp = torch.zeros((300,2))
        self.register_buffer('P',torch.cat((P,P_temp),1))
          
       #Initialising the cascades for the network.
        if self.opts.network_width == 'large':
            self.cascade_1 = LargeConvCascadeLayer(opts.dc_type)
            self.cascade_2 = LargeConvCascadeLayer(opts.dc_type)
            self.cascade_3 = LargeConvCascadeLayer(opts.dc_type)
            self.cascade_4 = LargeConvCascadeLayer(opts.dc_type)
        elif self.opts.network_width == 'normal':
            self.cascade_1 = ConvCascadeLayer(opts.dc_type)
            self.cascade_2 = ConvCascadeLayer(opts.dc_type)
            self.cascade_3 = ConvCascadeLayer(opts.dc_type)
            self.cascade_4 = ConvCascadeLayer(opts.dc_type)

    def forward(self, b, AQ):
        #Initialise the estimate by solving for b using the first 16 signals and using some regularisation
        #AQ = [B,30,47], b = [B,X,Y,Z,30,1] ---> AQ_Tb = [B, X, Y, Z, 47, 1]
        AQ_Tb = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),b)
        AQ_TAQ = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),AQ.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        #AQ_TAQ = [47,47] AQ_Tb = [B,X,Y,Z,47,1] ---> #c_hat = [B,X,Y,Z,18,1]
  
        ##AQ = [B,30,47] ---> AQ_TAQ_temp = [B, 18,18]
        AQ_TAQ_temp = torch.matmul(AQ[:,:,[i for i in range(16)]+[45,46]].transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1), AQ[:,:,[i for i in range(16)]+[45,46]].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        c_hat = torch.linalg.solve(AQ_TAQ_temp+0.01*torch.eye(18).to(b.device),AQ_Tb[:,:,:,:,[i for i in range(16)]+[45,46],:])
        
        #c_hat = [B,X,Y,Z,18,1] ---> c = [B,X,Y,Z,47,1]
        c = torch.zeros((c_hat.shape[0], c_hat.shape[1], c_hat.shape[2], c_hat.shape[3],47,1)).to(b.device)
        c[:,:,:,:,:16,:] = c_hat[:,:,:,:,:16,:]
        c[:,:,:,:,45:,:] = c_hat[:,:,:,:,16:,:]
        
        c_inp = c.transpose(1,4).squeeze()
        c_out = self.cascade_1(c_inp)
        c_hat = c_out.transpose(1,4)
        c_hat = c_hat.unsqueeze(5)
       
       #Data Consistency layer 1
        c = self.dc(c,c_hat, AQ_Tb,AQ_TAQ, b,1)

        c_inp = c.transpose(1,4).squeeze()
        c_out = self.cascade_2(c_inp)
        c_hat = c_out.transpose(1,4)
        c_hat = c_hat.unsqueeze(5)
        c_hat = self.res_con(c_hat,c)

        #Data Consistency Layer 2
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,2)
        
        c_inp = c.transpose(1,4).squeeze()
        c_out = self.cascade_3(c_inp)
        c_hat = c_out.transpose(1,4)
        c_hat = c_hat.unsqueeze(5)

        c_hat = self.res_con(c_hat,c)
        #Pass the data through the final data consistency layer
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,3)

        c_inp = c.transpose(1,4).squeeze()
        c_out = self.cascade_4(c_inp)
        c_hat = c_out.transpose(1,4)
        c_hat = c_hat.unsqueeze(5)
        c_hat = self.res_con(c_hat,c)
        
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,4)
        

        return c

    def dc(self, c, w, AQ_Tb, AQ_TAQ, b,n): 
        c = c[:,1:-1,1:-1,1:-1,:,:]
        if self.opts.dc_type == 'CSD':
            for i in range(3):
                #c=[B,X,Y,X,47,1], ---> L = [B,X,Y,Z,300,47]
                L = self.L_update(self.sampling_directions, c, 0,self.P, True, self.alpha) 
                
                #L = [B,X,Y,Z,300,47] ---> L_TL = [B,X,Y,Z,47,47]
                L_TL = torch.matmul(L.transpose(4,5),L).to(b.device)
            
                # AQ_TAQ = [B, 47,47], L_TL = [B,X,Y,Z,47,47], AQ_Tb = [B,X,Y,Z,47,1], w = [B,X,Y,Z,47,1] ---> c = [B,X,Y,Z,47,1]
                c = torch.linalg.solve(AQ_TAQ+(self.neg_reg)*L_TL+self.deep_reg*torch.eye(47).to(b.device),AQ_Tb[:,n:-n,n:-n,n:-n,:,:] + self.deep_reg*w)

        elif self.opts.dc_type == 'FOD_sig':
            A_tmp = AQ_TAQ+self.deep_reg*torch.matmul(self.P.transpose(0,1),self.P)
            b_tmp = AQ_Tb[:,n:-n,n:-n,n:-n,:,:]+self.deep_reg*torch.matmul(self.P.transpose(0,1),w)
            c = torch.linalg.solve(A_tmp, b_tmp)

        return c

    def res_con(self,c_hat,c):
        if self.opts.dc_type == 'FOD_sig':
            c_hat = c_hat+torch.matmul(self.P, c[:,1:-1,1:-1,1:-1,:,:])
            c_hat = F.relu(c_hat)
        elif self.opts.dc_type == 'CSD':
            c_hat = c_hat + c[:,1:-1,1:-1,1:-1,:,:]
        
        return c_hat
    
    def L_update(self, sampling_directions, c, tau,P, soft, alpha):
        '''
        The L update, takes the sampling directions - in most cases this will be the 300 pre defined directions which have been taken from 
        the MRTRIX3 code as well as the order of coefficients, the current estimate of c and the theresholding value tau and calculate the new updated
        L matrix. This only works when the input c is a single vector, and therefore isn't useful when applied to batched data - some changes have to be made
        elsewhere in the code to enable its use with batch data. ***This code has been updated to include the two volume fractions as per the 
        model based code***
        '''
        # alpha = 10
        #c=[B,X,Y,X,47,1], P = [300,47] ---> PC = [B,X,Y,Z,300,1]
        Pc = torch.matmul(P,c.float())
        
        
        if soft == False:
            #P = construct_sh_basis(sampling_directions, order)
            #L = torch.zeros((P.shape[0],P.shape[1])).double()
            #Pc = torch.matmul(P,c.double())
            #May need updating to incorperate batches
            indicies = []
            for i in range(len(Pc)):
                if Pc[i] < tau:
                    indicies.append(i)
            L = P[indicies,:]
        else:
            # P = [300,47], Pc = [B,X,Y,Z,300,1] ---> L = [B,X,Y,Z,300,47]
            L = torch.mul(P,torch.tile(torch.sigmoid(alpha*(tau - Pc)),(1,1,1,1,1,47)))

        
           
        
        return L
  






