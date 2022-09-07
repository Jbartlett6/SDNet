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

class FCCascadeLayer(nn.Module):
    """Cascade Layer"""

    def __init__(self):
        super().__init__()
        self.casc = nn.Sequential(nn.Linear(1269, 512),
                                  nn.BatchNorm1d(512),  
                                  nn.ReLU(inplace=True),  
                                  nn.Linear(512,512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512,512),
                                  nn.BatchNorm1d(512),  
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512,512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512,47))

    def forward(self, x):
        return self.casc(x)

class FCNet(nn.Module):
    def __init__(self, device, lambda_deep, lambda_neg,alpha, opts):
        super(FCNet, self).__init__()

        #Data consistency term parameters:
        self.register_buffer('lambda_deep', torch.tensor(lambda_deep))
        self.register_buffer('lambda_neg', torch.tensor(lambda_neg))
        self.register_buffer('alpha', torch.tensor(alpha))
        #self.lambda_deep = torch.nn.Parameter(torch.tensor(0.5))

        self.sampling_directions = torch.load(os.path.join('utils/300_predefined_directions.pt'))
        
        self.order = 8 
        self.device = device
        P = util.construct_sh_basis(self.sampling_directions, self.order)
        P_temp = torch.zeros((300,2))
        self.register_buffer('P',torch.cat((P,P_temp),1))
          
        #Initialising the cascades for the network.
        self.cascade_1 = FCCascadeLayer()
        self.cascade_2 = FCCascadeLayer()
        self.cascade_3 = FCCascadeLayer()
        self.cascade_4 = FCCascadeLayer()
        # self.convcasc1 = torch.nn.Conv3d(47,47,(3,3,3))
        # self.convcasc2 = torch.nn.Conv3d(47,47,(3,3,3))
        # self.convcasc3 = torch.nn.Conv3d(47,47,(3,3,3))

    
        

    def forward(self, b, AQ):
      

        #Initialise the estimate by solving for b using the first 16 signals and using some regularisation
        #AQ = [B,30,47], b = [B,X,Y,Z,30,1] ---> AQ_Tb = [B, X, Y, Z, 47, 1]
        AQ_Tb = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),b)
        AQ_TAQ = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),AQ.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        #AQ_TAQ = [47,47] AQ_Tb = [B,X,Y,Z,47,1] ---> #c_hat = [B,X,Y,Z,18,1]
        #Solve the first 16 spherical harmonic coefficients for the white matter FOD as well as the grey and white matter volume fractions. This
        #is used as an initialisation for the algorithm. The high order terms are ommited as they are in CSD - my logic being that they may be more
        #difficult to solve when limited data is available.
        ##AQ = [B,30,47] ---> AQ_TAQ_temp = [B, 18,18]
        AQ_TAQ_temp = torch.matmul(AQ[:,:,[i for i in range(16)]+[45,46]].transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1), AQ[:,:,[i for i in range(16)]+[45,46]].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        c_hat = torch.linalg.solve(AQ_TAQ_temp+0.01*torch.eye(18).to(b.device),AQ_Tb[:,:,:,:,[i for i in range(16)]+[45,46],:])
        #c_hat = torch.linalg.solve(self.AQ_TAQ[[i for i in range(16)]+[45,46],[i for i in range(16)]+[45,46]]+0.01*torch.eye(18).to(b.device),AQ_Tb[:,:,:,:,[i for i in range(16)]+[45,46],:])
        #c_hat = [B,X,Y,Z,18,1] ---> c = [B,X,Y,Z,47,1]
        c = torch.zeros((c_hat.shape[0], c_hat.shape[1], c_hat.shape[2], c_hat.shape[3],47,1)).to(b.device)
        c[:,:,:,:,:16,:] = c_hat[:,:,:,:,:16,:]
        c[:,:,:,:,45:,:] = c_hat[:,:,:,:,16:,:]
        


        # Pass data through cascade layer 1 (need to squeeze due to the requirements of the linear layer)
        # c = [B,X,Y,Z,47,1] ---> c_hat = [B, X-2, Y-2, Z-2, 47, 1]
        c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(b.device)
        for i in range(1,c.shape[1]-1):
            for j in range(1,c.shape[2]-1):
                for k in range(1,c.shape[3]-1):
                    inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
                    #inp = [B,1215]
                    inp = inp.reshape(c.shape[0],-1)
                    w = self.cascade_1(inp)
                    c_hat[:,i-1,j-1,k-1,:,0] = w
        

        # c = c.transpose(1,4)
        # c = self.convcasc1(c)
        # c_hat = c.transpose(1,4)
       
       #Data Consistency layer 1
        c = self.dc(c,c_hat, AQ_Tb,AQ_TAQ, b,1)
                
        c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(b.device)
        for i in range(1,c.shape[1]-1):
            for j in range(1,c.shape[2]-1):
                for k in range(1,c.shape[3]-1):
                    inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
                   
                    #inp = [B,1215]
                    inp = inp.reshape(b.shape[0],-1)
                    w = self.cascade_2(inp)
                    c_hat[:,i-1,j-1,k-1,:,0] = w

        c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
        
        #Data Consistency Layer 2
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,2)
    
        # c = c.transpose(1,4)
        # c = self.convcasc2(c)
        # c_hat = c.transpose(1,4)

      # Pass data through cascade layer 3 (need to squeeze due to the requirements of the linear layer)
        c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(self.device)
        for i in range(1,c.shape[1]-1):
            for j in range(1,c.shape[2]-1):
                for k in range(1,c.shape[3]-1):
                    inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
                    #inp = [B,1215]
                    inp = inp.reshape(b.shape[0],-1)
                    w = self.cascade_3(inp)
                    c_hat[:,i-1,j-1,k-1,:,0] = w
        
        c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
        # c = c.transpose(1,4)
        # c = self.convcasc3(c)
        # c_hat = c.transpose(1,4)
        
        #Pass the data through the final data consistency layer
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,3)

        c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(self.device)
        for i in range(1,c.shape[1]-1):
            for j in range(1,c.shape[2]-1):
                for k in range(1,c.shape[3]-1):
                    inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
                    #inp = [B,1215]
                    inp = inp.reshape(b.shape[0],-1)
                    w = self.cascade_4(inp)
                    c_hat[:,i-1,j-1,k-1,:,0] = w
        
        c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
        c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,4)
        

        return c

    def dc(self, c, w, AQ_Tb, AQ_TAQ, b,n): 
        c = c[:,1:-1,1:-1,1:-1,:,:]
        for i in range(3):
            #Need to update L for every c in the batch, therefore we create a template then loop through the
            #over the batch dimension and update for each given c.The first estimate for the matrix L is calculated using the previous estimate for c
            #L = torch.zeros((c.shape[0],300,45))
             
            #for j in range(c.shape[0]):
                #L[j,:,:] = self.L_update(self.sampling_directions, c[j,:,:], 0,self.P, True, 10)
            #c=[B,X,Y,X,47,1], ---> L = [B,X,Y,Z,300,47]
            L = self.L_update(self.sampling_directions, c, 0,self.P, True, self.alpha)
            #Try to make this more efficient - adapt L_update so the device change isn't needed every loop/ the bmm without device doesn't occur 
            
            #L = [B,X,Y,Z,300,47] ---> L_TL = [B,X,Y,Z,47,47]
            L_TL = torch.matmul(L.transpose(4,5),L).to(b.device)
            

            
            #Update these values so they reflect what was in the MRtrix code and can be adapted. Addition should work with the identity matrix due to broadcasting - 
            #note that the identity will be padded to the left, i.e. in the batch dimension, the matrix is then repeated in this dimension. I have added the 
            #additional scaling of the deep regularisation term in order to ensure that the data term and this deep reg term are balanced (approximately on the 
            # same scale - this discrepency is caused by the different scales of the signals and the spherical harmonic coefficients) 
            # AQ_TAQ = [B, 47,47], L_TL = [B,X,Y,Z,47,47], AQ_Tb = [B,X,Y,Z,47,1], w = [B,X,Y,Z,47,1] ---> c = [B,X,Y,Z,47,1]

            c = torch.linalg.solve(AQ_TAQ+(self.lambda_neg)*L_TL+self.lambda_deep*torch.eye(47).to(b.device),AQ_Tb[:,n:-n,n:-n,n:-n,:,:] + self.lambda_deep*w)
        # print(F.mse_loss(torch.matmul(self.AQ,c),b[:,n:-n,n:-n,n:-n,:,:]))
        # print(self.lambda_deep*F.mse_loss(w,c))      
        # print('dc loop')
        # toc()
        return c
    
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
  






