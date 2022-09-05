import time
import sys
import os 
sys.path.append(os.path.join(sys.path[0],'models'))
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
import scipy.special 



#Tic toc functions useful for timing
#############################################################################
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def ACC(out,gt):
    '''
    A function to calculate the Angular Correlation Coefficient between the output of the network and the 
    ground truth data. The data should have dimensions gt = [B,SH], out = [B, 1,1,1,SH,1]. If the network is adapted to output 
    patches of a different shape, i.e. patches in the spatial domain. - Although shouldn't take major canges as the 
    shape of gt should match that of out - only the axis for which operations are performed over will require change.
    '''
    out = out.to('cpu')
    gt = gt.to('cpu')
    out = out.squeeze().detach().numpy()
    num = np.multiply(out[:,1:45],gt[:,1:45])
    num = torch.sum(num, axis = 1)
    #Calculating the Frobenius norm of each of the 
    denum = np.linalg.norm(out[:,1:45],axis = 1) * np.linalg.norm(gt[:,1:45], axis=1)
    return num/denum

   
def bvec_extract(subject): 
    '''
    Input:
        subj (str) - The subject whose bvectors you want to extract.
        path (str) - The path to where the data is located (subjects location).
    Desc:
        A function to extract the bvectors from the file they are located in and to return them as a list of vectors.
    '''
    path = '/media/duanj/F/joe/hcp_2/'+subject+'/T1w/Diffusion/undersampled_fod/bvecs'
    
    bvecs = open(path ,'r')

    bvecs_str = bvecs.read()
    bvecsxyz = bvecs_str.split('\n')
    
    xvals = [float(x) for x in bvecsxyz[0].split()]
    yvals = [float(y) for y in bvecsxyz[1].split()]
    zvals = [float(z) for z in bvecsxyz[2].split()]

    bvecs = [[-xvals[i], yvals[i], zvals[i]] for i in range(len(xvals))]
    return torch.tensor(bvecs)

def bval_extract(data_dir, subject, sub_dir):
    '''
    Input:
        subj (str) - The subject whose bvals you want to extract.
        sub_dir (str) - The sub directory within the subject's folder where the bvalues are located. This should just be the folder name.
        e.g. for /Diffusion/undersampled_fod,bvals input 'undersampled_fod' as the argument. 
    
    Desc:
        A function to extract the bvalues from the file they are located in and to return them as a list of values. The function assumes that 
        the subject is found in the hcp_2 directory. The fully sampled bvalues and bvectors are in the T1w diffusion folder, then any undersampled
        bvalues and bvectors are found in the subdirectory. Such subdirectory bvalues and bvectors are specified by the sub_dir argument.
    '''
    if sub_dir == None:
        path = os.path.join(data_dir, str(subject), 'T1w', 'Diffusion','bvals')
    else:
        path = os.path.join(data_dir, str(subject), 'T1w', 'Diffusion',sub_dir,'bvals')
    

    bvals = open(path, 'r')
    
    bvals_str = bvals.read()
    bvals = [int(b) for b in bvals_str.split()]
    
    return bvals

def ss_sph_coords(bvecs):
    '''
    A function to convert the bvectors from cartesian coordiantes to spherical coordinates- they can then be used to create
    the spherical harmonic basis matrix A (as the scipy function takes spherical coordiantes as input). The first column is the azimuth, 
    the second column is the elevation - this is the same convention as is used in the MRTRIX code.
    '''
    az = torch.atan2(bvecs[:,1], bvecs[:,0]).reshape((-1,1))
    pol = torch.atan2(torch.sqrt(bvecs[:,0]**2+bvecs[:,1]**2),bvecs[:,2] ).reshape((-1,1))
    
    return torch.cat((az,pol),1)

def real_sh_eval(m,n,az,el):
    '''
    Evaluates the real spherical harmonic coefficients for m = order/phase, n = degree.
    '''
    if n%2:
        sh = 0
    elif m<0:
        if m%2 == True:
            pow = 0
        else:
            pow = 1
            

        sh = ((-1)**pow)*math.sqrt(2)*torch.imag(scipy.special.sph_harm(m, n, az, el))
    elif m == 0:
        sh = scipy.special.sph_harm(m, n, az, el)
    elif m>0:
        sh = math.sqrt(2)*torch.real(scipy.special.sph_harm(m, n, az, el))
    
    return sh

def construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals):
        '''
        Takes the bvector gradient directions and the spherical harmonic coefficients max
        order and creates a matrix containing said spherical harmonic bases. Has the same 
        number of rows as there are signals (or bvectors) and the same number of columns as 
        there are spherical harmonic bases (including those for both csd and grey matter
        which are both essentially scalar volume fractions). The function requires the 
        response function parameters of g_wm, g_gm and g_csf in order to scale each set of 
        SH bases correctly. The matrix created by this function is essentially AQ rather than 
        just A alone. 
        '''
        ##To do - clean up the coding for his - I don't think this makes a huge amount of sense to do it this way.
        A_cumulative = torch.zeros((bvecs_sph.shape[0],1)).float()
        for l in range(order+1):
            if l%2 == 0:
                A_temp = torch.zeros((bvecs_sph.shape[0],2*l+1))
                for k in range(2*l+1):
                    for n in range(bvecs_sph.shape[0]):
                        A_temp[n,k] = real_sh_eval(k-l,l,bvecs_sph[n,0],bvecs_sph[n,1])
                
                A_cumulative = torch.cat((A_cumulative,A_temp),1)
        
        A_temp = torch.zeros((bvecs_sph.shape[0],2))
        for n in range(bvecs_sph.shape[0]):
                    A_temp[n,-2:] = real_sh_eval(0,0,bvecs_sph[n,0],bvecs_sph[n,1])
        A_cumulative = torch.cat((A_cumulative,A_temp),1)

        A = A_cumulative[:,1:]
        
        G = torch.zeros((4,47))
        
        for i in range(4):
            temp = []
            g_ind = 0
            for l in range(0,8+1,2):
                for m in range(-l,l+1):
                    temp.append(g_wm[i,g_ind]/real_sh_eval(0,l,0,0) )
                g_ind += 1

            temp.append(g_gm[i]/real_sh_eval(0,0,0,0))
            temp.append(g_csf[i]/real_sh_eval(0,0,0,0))
            G[i,:] = torch.tensor(temp)

        for i in range(A.shape[0]):
            if bvals[i]<20:
                A[i] *= G[0,:]
            elif 800<bvals[i]<1200:
                A[i] *= G[1,:]
            elif 1800<bvals[i]<2200:
                A[i] *= G[2,:]
            elif 2800<bvals[i]<3200:
                A[i] *= G[3,:]  

        return A

def construct_sh_basis(bvecs_sph, order):
    '''
    Takes the bvector gradient directions and the spherical harmonic coefficients max
    order and creates a matrix containing said spherical harmonic bases. Has the same 
    number of rows as there are signals (or bvectors) and the same number of columns as 
    there are spherical harmonic bases.
    '''
    A_cumulative = torch.zeros((bvecs_sph.shape[0],1)).float()
    for l in range(order+1):
        if l%2 == 0:
            A_temp = torch.zeros((bvecs_sph.shape[0],2*l+1))
            for k in range(2*l+1):
                for n in range(bvecs_sph.shape[0]):
                    A_temp[n,k] = real_sh_eval(k-l,l,bvecs_sph[n,0],bvecs_sph[n,1])
            A_cumulative = torch.cat((A_cumulative,A_temp),1)
    return A_cumulative[:,1:]