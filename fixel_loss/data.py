import torch
import nibabel as nib
import os
import numpy as np


class FODPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        
        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),145,174,145,45))
        self.gt_tensor = torch.zeros((len(subject_list),145,174,145))
        self.wb_mask_tensor = torch.zeros((len(subject_list),145,174,145))
        self.ttgen_mask_tensor = torch.zeros((len(subject_list),145,174,145,5))


        #Loading the data into the data tensor
        print('Loading the FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir, subject,'T1w','Diffusion','wmfod.nii.gz')
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))
        self.affine = nifti.affine

        
        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir, subject,'T1w','Diffusion','fixel_directory','fixnet_targets','gt_threshold_fixels.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:] = torch.tensor(np.array(nifti.dataobj).astype(float))[:,:,:,0]

        
        #Loading the mask data into RAM
        print('Loading the whole brain and 5ttgen mask data into RAM')
        for i, subject in enumerate(subject_list):
            
            #Importing the whole brain mask
            path_wb = os.path.join(self.data_dir,subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            nifti_wb = nib.load(path_wb)
            self.wb_mask_tensor[i,:,:,:] = torch.tensor(np.array(nifti_wb.dataobj))
            
            #Importing the 5ttgen mask
            path_5ttgen = os.path.join(self.data_dir,subject,'T1w','5ttgen.nii.gz')
            nifti_5ttgen = nib.load(path_5ttgen)
            self.ttgen_mask_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti_5ttgen.dataobj))
           

        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

        #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        self.coords = grid[(self.ttgen_mask_tensor[:,:,:,:,0].to(bool) | self.ttgen_mask_tensor[:,:,:,:,1].to(bool) | self.ttgen_mask_tensor[:,:,:,:,2].to(bool)) & self.wb_mask_tensor.to(bool),:]
    

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
        input_signals = self.data_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3], :]
        
        target_fod = self.gt_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3]]
        
       
        return input_signals.float(), target_fod.float(), central_coords

class InferenceFODPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list,fod_path):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        
        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),145,174,145,47))
        self.gt_tensor = torch.zeros((len(subject_list),145,174,145))
        self.wb_mask_tensor = torch.zeros((len(subject_list),145,174,145))
        self.ttgen_mask_tensor = torch.zeros((len(subject_list),145,174,145,5))


        #Loading the data into the data tensor
        print('Loading the FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = fod_path
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))
        self.affine = nifti.affine

        #Loading the mask data into RAM
        print('Loading the whole brain and 5ttgen mask data into RAM')
        for i, subject in enumerate(subject_list):
            
            #Importing the whole brain mask
            path_wb = os.path.join(self.data_dir,subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            nifti_wb = nib.load(path_wb)
            self.wb_mask_tensor[i,:,:,:] = torch.tensor(np.array(nifti_wb.dataobj))
            
            #Importing the 5ttgen mask
            path_5ttgen = os.path.join(self.data_dir,subject,'T1w','5ttgen.nii.gz')
            nifti_5ttgen = nib.load(path_5ttgen)
            self.ttgen_mask_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti_5ttgen.dataobj))
           

        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

        #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        self.coords = grid[(self.ttgen_mask_tensor[:,:,:,:,0].to(bool) | self.ttgen_mask_tensor[:,:,:,:,1].to(bool) | self.ttgen_mask_tensor[:,:,:,:,2].to(bool)) & self.wb_mask_tensor.to(bool),:]
    

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
        input_signals = self.data_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3], :]
        
       
        return input_signals.float(), central_coords

