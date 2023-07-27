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
import utils.data as data

class ceblock(nn.Module):
    def __init__(self, num_coeff):
        super(ceblock, self).__init__()
        self.l_0 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.bn_0 = torch.nn.BatchNorm1d(1024)
        self.glu_0 = torch.nn.GLU(dim=1)
        self.l_1 = torch.nn.Linear(in_features=512, out_features=512)
        self.bn_1 = torch.nn.BatchNorm1d(512)
        self.glu_1 = torch.nn.GLU(dim=1)
        self.l_2 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_2 = torch.nn.BatchNorm1d(512)
        self.glu_2 = torch.nn.GLU(dim=1)
        self.l_3 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_3 = torch.nn.BatchNorm1d(512)
        self.glu_3 = torch.nn.GLU(dim=1)
        self.l_4 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_4 = torch.nn.BatchNorm1d(512)
        self.glu_4 = torch.nn.GLU(dim=1)
        self.pred = torch.nn.Linear(in_features=256, out_features=num_coeff)

    def forward(self, x):
        x = self.l_0(x)
        x = self.bn_0(x)
        x = self.glu_0(x)
        x = self.l_1(x)
        x = self.bn_1(x)
        x = self.glu_1(x)
        x = self.l_2(x)
        x = self.bn_2(x)
        x = self.glu_2(x)
        x = self.l_3(x)
        x = self.bn_3(x)
        x = self.glu_3(x)
        x = self.l_4(x)
        x = self.bn_4(x)
        x = self.glu_4(x)
        x = self.pred(x)
        return x

class Sep9ResdiualDeeperBN_1(nn.Module):
    """Baseline moduel
    """
    def __init__(self):
        """Construct a Resnet-based generator
        """
        super(Sep9ResdiualDeeperBN_1, self).__init__()
        self.conv3d1 = torch.nn.Conv3d(in_channels=47, out_channels=256, kernel_size=3)
        self.bn3d1 = torch.nn.BatchNorm3d(256)
        self.glu3d1 = torch.nn.GLU(dim=1)

        self.conv3d2 = torch.nn.Conv3d(in_channels=128, out_channels=512, kernel_size=3)
        self.bn3d2 = torch.nn.BatchNorm3d(512)
        self.glu3d2 = torch.nn.GLU(dim=1)

        self.conv3d3 = torch.nn.Conv3d(in_channels=256, out_channels=1024, kernel_size=3)
        self.bn3d3 = torch.nn.BatchNorm3d(1024)
        self.glu3d3 = torch.nn.GLU(dim=1)

        self.conv3d4 = torch.nn.Conv3d(in_channels=512, out_channels=2048, kernel_size=3)
        self.bn3d4 = torch.nn.BatchNorm3d(2048)
        self.glu3d4 = torch.nn.GLU(dim=1)

        self.joint_linear = torch.nn.Linear(in_features=1024, out_features=2048)
        self.joint_bn = torch.nn.BatchNorm1d(2048)
        self.joint_glu = torch.nn.GLU(dim=1)

        self.l0_pred = ceblock(num_coeff=250)
        self.l2_pred = ceblock(num_coeff=1250)
        self.l4_pred = ceblock(num_coeff=2250)
        self.l6_pred = ceblock(num_coeff=3250)
        self.l8_pred = ceblock(num_coeff=4250)
        self.iso_pred = ceblock(num_coeff = 500)

    def forward(self, fodlr):
        x = self.conv3d1(fodlr)
        x = self.bn3d1(x)
        x = self.glu3d1(x)

        x = self.conv3d2(x)
        x = self.bn3d2(x)
        x = self.glu3d2(x)

        x = self.conv3d3(x)
        x = self.bn3d3(x)
        x = self.glu3d3(x)

        x = self.conv3d4(x)
        x = self.bn3d4(x)
        x = self.glu3d4(x)

        x = x.squeeze()
        x = self.joint_linear(x)
        x = self.joint_bn(x)
        joint = self.joint_glu(x)

        x = self.l0_pred(joint)
        l0_residual = x[:, :1].reshape((-1,1,5,5,5))
        l0_scale = F.sigmoid(x[:, 1:]).reshape((-1,1,5,5,5))

        x = self.l2_pred(joint)
        l2_residual = x[:, :5].reshape((-1,5,5,5,5))
        l2_scale = F.sigmoid(x[:, 5:]).reshape((-1,5,5,5,5))

        x = self.l4_pred(joint)
        l4_residual = x[:, :9].reshape((-1,9,5,5,5))
        l4_scale = F.sigmoid(x[:, 9:]).reshape((-1,9,5,5,5))

        x = self.l6_pred(joint)
        l6_residual = x[:, :13].reshape((-1,13,5,5,5))
        l6_scale = F.sigmoid(x[:, 13:]).reshape((-1,13,5,5,5))

        x = self.l8_pred(joint)
        l8_residual = x[:, :17].reshape((-1,17,5,5,5))
        l8_scale = F.sigmoid(x[:, 17:]).reshape((-1,17,5,5,5))

        x = self.iso_pred(joint)
        iso_residual = x[:, :2].reshape((-1,2,5,5,5))
        iso_scale = F.sigmoid(x[:, 2:]).reshape((-1,2,5,5,5))

        residual = torch.cat([l0_residual, l2_residual, l4_residual, l6_residual, l8_residual, iso_residual], dim=4)
        scale = torch.cat([l0_scale, l2_scale, l4_scale, l6_scale, l8_scale, iso_scale], dim=4)

        fodpred = residual * scale + fodlr[:, :, 2:7, 2:7, 2:7]

        return fodpred



class Sep9ResdiualDeeperBN_2(nn.Module):
    """Baseline moduel
    """
    def __init__(self):
        """Construct a Resnet-based generator
        """
        super(Sep9ResdiualDeeperBN_2, self).__init__()
        self.conv3d1 = torch.nn.Conv3d(in_channels=47, out_channels=256, kernel_size=3)
        self.bn3d1 = torch.nn.BatchNorm3d(256)
        self.glu3d1 = torch.nn.GLU(dim=1)

        self.conv3d2 = torch.nn.Conv3d(in_channels=128, out_channels=512, kernel_size=3)
        self.bn3d2 = torch.nn.BatchNorm3d(512)
        self.glu3d2 = torch.nn.GLU(dim=1)

        self.conv3d3 = torch.nn.Conv3d(in_channels=256, out_channels=1024, kernel_size=3)
        self.bn3d3 = torch.nn.BatchNorm3d(1024)
        self.glu3d3 = torch.nn.GLU(dim=1)

        self.conv3d4 = torch.nn.Conv3d(in_channels=512, out_channels=2048, kernel_size=3)
        self.bn3d4 = torch.nn.BatchNorm3d(2048)
        self.glu3d4 = torch.nn.GLU(dim=1)

        self.joint_linear = torch.nn.Linear(in_features=1024, out_features=2048)
        self.joint_bn = torch.nn.BatchNorm1d(2048)
        self.joint_glu = torch.nn.GLU(dim=1)

        self.l0_pred = ceblock(num_coeff=2)
        self.l2_pred = ceblock(num_coeff=10)
        self.l4_pred = ceblock(num_coeff=18)
        self.l6_pred = ceblock(num_coeff=26)
        self.l8_pred = ceblock(num_coeff=34)
        self.iso_pred = ceblock(num_coeff = 4)

    def forward(self, fodlr):
        x = self.conv3d1(fodlr)
        x = self.bn3d1(x)
        x = self.glu3d1(x)

        x = self.conv3d2(x)
        x = self.bn3d2(x)
        x = self.glu3d2(x)

        x = self.conv3d3(x)
        x = self.bn3d3(x)
        x = self.glu3d3(x)

        x = self.conv3d4(x)
        x = self.bn3d4(x)
        x = self.glu3d4(x)

        x = x.squeeze()
        x = self.joint_linear(x)
        x = self.joint_bn(x)
        joint = self.joint_glu(x)

        x = self.l0_pred(joint)
        l0_residual = x[:, :1]
        l0_scale = F.sigmoid(x[:, 1:])

        x = self.l2_pred(joint)
        l2_residual = x[:, :5]
        l2_scale = F.sigmoid(x[:, 5:])

        x = self.l4_pred(joint)
        l4_residual = x[:, :9]
        l4_scale = F.sigmoid(x[:, 9:])

        x = self.l6_pred(joint)
        l6_residual = x[:, :13]
        l6_scale = F.sigmoid(x[:, 13:])

        x = self.l8_pred(joint)
        l8_residual = x[:, :17]
        l8_scale = F.sigmoid(x[:, 17:])

        x = self.iso_pred(joint)
        iso_residual = x[:, :2]
        iso_scale = F.sigmoid(x[:, 2:])

        residual = torch.cat([l0_residual, l2_residual, l4_residual, l6_residual, l8_residual, iso_residual], dim=1)
        scale = torch.cat([l0_scale, l2_scale, l4_scale, l6_scale, l8_scale, iso_scale], dim=1)

        fodpred = residual * scale + fodlr[:, :, 4, 4, 4]

        return fodpred




class FCNet(nn.Module):
    def __init__(self, device, lambda_deep, lambda_neg,alpha):
        super(FCNet, self).__init__()

        #Data consistency term parameters:
        self.register_buffer('lambda_deep', torch.tensor(lambda_deep))
        self.register_buffer('lambda_neg', torch.tensor(lambda_neg))
        self.register_buffer('alpha', torch.tensor(alpha))
        #self.lambda_deep = torch.nn.Parameter(torch.tensor(0.5))

        self.sampling_directions = torch.load('/home/jxb1336/code/Project_1: HARDI_Recon/FOD-REG_NET/300_predefined_directions.pt')
        self.order = 8 
        self.device = device
        P = util.construct_sh_basis(self.sampling_directions, self.order)
        P_temp = torch.zeros((300,2))
        self.register_buffer('P',torch.cat((P,P_temp),1))
          
        #Initialising the cascades for the network.
        self.cascade_1 = Sep9ResdiualDeeperBN_1()
        self.cascade_2 = Sep9ResdiualDeeperBN_2()

    
        

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
        #Taking the transpose of c so that the second dimension is the number of channels.
        c_in = c.squeeze().transpose(1,4)


        # Pass data through cascade layer 1 (need to squeeze due to the requirements of the linear layer)
        # c = [B,X,Y,Z,47,1] ---> c_hat = [B, X-2, Y-2, Z-2, 47, 1]
        # c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(b.device)
        # for i in range(1,c.shape[1]-1):
        #     for j in range(1,c.shape[2]-1):
        #         for k in range(1,c.shape[3]-1):
        #             inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
        #             #inp = [B,1215]
        #             inp = inp.reshape(c.shape[0],-1)
        #             w = self.cascade_1(inp)
        #             c_hat[:,i-1,j-1,k-1,:,0] = w

        c_hat = self.cascade_1(c_in)
        #c_hat = [B,5,5,5,47]

        # c = c.transpose(1,4)
        # c = self.convcasc1(c)
        # c_hat = c.transpose(1,4)
       
       #Data Consistency layer 1
        c = self.dc(c[:,2:7,2:7,2:7,:,:].squeeze().transpose(0,1),c_hat, AQ_Tb,AQ_TAQ, b,4)
                
    #     c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(b.device)
    #     for i in range(1,c.shape[1]-1):
    #         for j in range(1,c.shape[2]-1):
    #             for k in range(1,c.shape[3]-1):
    #                 inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
                   
    #                 #inp = [B,1215]
    #                 inp = inp.reshape(b.shape[0],-1)
    #                 w = self.cascade_2(inp)
    #                 c_hat[:,i-1,j-1,k-1,:,0] = w

    #     c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
        
    #     #Data Consistency Layer 2
    #     c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,2)
    
    #     # c = c.transpose(1,4)
    #     # c = self.convcasc2(c)
    #     # c_hat = c.transpose(1,4)

    #   # Pass data through cascade layer 3 (need to squeeze due to the requirements of the linear layer)
    #     c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(self.device)
    #     for i in range(1,c.shape[1]-1):
    #         for j in range(1,c.shape[2]-1):
    #             for k in range(1,c.shape[3]-1):
    #                 inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
    #                 #inp = [B,1215]
    #                 inp = inp.reshape(b.shape[0],-1)
    #                 w = self.cascade_3(inp)
    #                 c_hat[:,i-1,j-1,k-1,:,0] = w
        
    #     c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
    #     # c = c.transpose(1,4)
    #     # c = self.convcasc3(c)
    #     # c_hat = c.transpose(1,4)
        
    #     #Pass the data through the final data consistency layer
    #     c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,3)

    #     c_hat = torch.zeros((c.shape[0], c.shape[1]-2, c.shape[2]-2, c.shape[3]-2, c.shape[4], c.shape[5])).to(self.device)
    #     for i in range(1,c.shape[1]-1):
    #         for j in range(1,c.shape[2]-1):
    #             for k in range(1,c.shape[3]-1):
    #                 inp = c[:,i-1:i+2, j-1:j+2, k-1:k+2, :,:].squeeze()
    #                 #inp = [B,1215]
    #                 inp = inp.reshape(b.shape[0],-1)
    #                 w = self.cascade_4(inp)
    #                 c_hat[:,i-1,j-1,k-1,:,0] = w
        
    #     c_hat = c_hat+c[:,1:-1,1:-1,1:-1,:,:]
    #     c = self.dc(c,c_hat, AQ_Tb, AQ_TAQ, b,4)
        

        return c

    def dc(self, c, w, AQ_Tb, AQ_TAQ, b,n): 
        #c = c[:,1:-1,1:-1,1:-1,:,:]
        c = c.transpose(0,1).unsqueeze(2).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        w = w.unsqueeze(2).unsqueeze(1).unsqueeze(1).unsqueeze(1)

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

            c = torch.linalg.solve(AQ_TAQ+self.lambda_neg*L_TL+self.lambda_deep*torch.eye(47).to(b.device),AQ_Tb[:,4:5,4:5,4:5,:,:] + self.lambda_deep*w)
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
            # P = [300,47], Pc = [B,300,1] ---> L = [B,300,47]
            L = torch.mul(P,torch.tile(torch.sigmoid(alpha*(tau - Pc)),(1,1,1,1,1,47)))

        
           
        
        return L