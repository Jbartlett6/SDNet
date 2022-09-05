import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.special
import os 
import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
import util 


class DWIPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list, inference):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.inference = inference
        
        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),145,174,145,30))
        self.gt_tensor = torch.zeros((len(subject_list),145,174,145,47))
        self.AQ_tensor = torch.zeros((len(subject_list),30,47))
        self.mask_tensor = torch.zeros((len(subject_list),145,174,145))


        #Loading the data into the data tensor
        print('Loading the signal data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','normalised_data.nii.gz')
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))
        
        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            nifti = nib.load(path)
            self.mask_tensor[i,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting bvectors from folders:
            bvecs = util.bvec_extract(subject)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            order = 8

            #Extracting bvalues:
            bvals = util.bval_extract(self.data_dir, subject, 'undersampled_fod')
            #White matter response function extraaction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/wm_response.txt', 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(4,6)
            for j in range(4):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/gm_response.txt', 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/csf_response.txt', 'r') as txt:
                x = txt.read()
            g_csf = [float(resp) for resp in x.split('\n')[2:-1]]    
        
            
            self.AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals)
        
        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

            #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        self.coords = grid[self.mask_tensor.to(bool),:]
    

    def __len__(self):
        #return int(torch.sum(torch.tensor(self.mask)))
        return self.coords.shape[0]
    

    def __getitem__(self, idx):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        central_coords = self.coords[idx,:]       
        
                #Obtains the signals and the target FOD. The signals which are kept are determined by the keep list.
        input_signals = self.data_tensor[central_coords[0],central_coords[1]-4:central_coords[1]+5, central_coords[2]-4:central_coords[2]+5, central_coords[3]-4:central_coords[3]+5, :]
        
        target_fod = self.gt_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3], :]
        
        AQ = self.AQ_tensor[central_coords[0],:,:]
        if self.inference:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), central_coords
        else:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float()

class FODPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list, inference):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        
        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),145,174,145,47))
        self.gt_tensor = torch.zeros((len(subject_list),145,174,145,47))
        self.AQ_tensor = torch.zeros((len(subject_list),30,47))
        self.mask_tensor = torch.zeros((len(subject_list),145,174,145))


        #Loading the data into the data tensor
        print('Loading the signal data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','undersampled_fod.nii.gz')
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))
        
        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            nifti = nib.load(path)
            self.mask_tensor[i,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting bvectors from folders:
            bvecs = util.bvec_extract(subject)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            order = 8

            #Extracting bvalues:
            bvals = util.bval_extract(self.data_dir, subject, 'undersampled_fod')
            #White matter response function extraaction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/wm_response.txt', 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(4,6)
            for j in range(4):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/gm_response.txt', 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/csf_response.txt', 'r') as txt:
                x = txt.read()
            g_csf = [float(resp) for resp in x.split('\n')[2:-1]]    
        
            
            self.AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals)
        
        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

            #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        self.coords = grid[self.mask_tensor.to(bool),:]
    

    def __len__(self):
        #return int(torch.sum(torch.tensor(self.mask)))
        return self.coords.shape[0]
    

    def __getitem__(self, idx):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        central_coords = self.coords[idx,:]       
        
                #Obtains the signals and the target FOD. The signals which are kept are determined by the keep list.
        input_signals = self.data_tensor[central_coords[0],central_coords[1]-4:central_coords[1]+5, central_coords[2]-4:central_coords[2]+5, central_coords[3]-4:central_coords[3]+5, :]
        
        target_fod = self.gt_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3], :]
        
        AQ = self.AQ_tensor[central_coords[0],:,:]
        if self.inference:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), central_coords
        else:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float()



class ExperimentPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list, inference):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.inference = inference
        
        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),79,87,97,30))
        self.gt_tensor = torch.zeros((len(subject_list),79,87,97,47))
        self.AQ_tensor = torch.zeros((len(subject_list),30,47))
        self.mask_tensor = torch.zeros((len(subject_list),79,87,97))


        #Loading the data into the data tensor
        print('Loading the signal data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','normalised_data.nii.gz')
            nifti = nib.load(path)
            #Shape = [62, 70, 80, :]
            self.data_tensor[i,4:66,4:74,4:84,:] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124,:]

        
        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,4:66,4:74,4:84,:] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124,:]

        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            nifti = nib.load(path)
            self.mask_tensor[i,4:66,4:74,4:84] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124]

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting bvectors from folders:
            bvecs = util.bvec_extract(subject)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            order = 8

            #Extracting bvalues:
            bvals = util.bval_extract(self.data_dir, subject, 'undersampled_fod')
            #White matter response function extraaction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/wm_response.txt', 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(4,6)
            for j in range(4):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/gm_response.txt', 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open('/media/duanj/F/joe/hcp_2/100206/T1w/Diffusion/undersampled_fod/csf_response.txt', 'r') as txt:
                x = txt.read()
            g_csf = [float(resp) for resp in x.split('\n')[2:-1]]    
        
            
            self.AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals)
        
        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

            #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        self.coords = grid[self.mask_tensor.to(bool),:]
    

    def __len__(self):
        #return int(torch.sum(torch.tensor(self.mask)))
        return self.coords.shape[0]
    

    def __getitem__(self, idx):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        central_coords = self.coords[idx,:]       
        
                #Obtains the signals and the target FOD. The signals which are kept are determined by the keep list.
        input_signals = self.data_tensor[central_coords[0],central_coords[1]-4:central_coords[1]+5, central_coords[2]-4:central_coords[2]+5, central_coords[3]-4:central_coords[3]+5, :]
        
        target_fod = self.gt_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3], :]
        
        AQ = self.AQ_tensor[central_coords[0],:,:]
        if self.inference:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), central_coords
        else:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float()
