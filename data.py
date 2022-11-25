import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import os 
import sys
import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
sys.path.append(os.path.join(sys.path[0],'utils'))
import util 
import h5py



class DWIPatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, subject_list, inference, opts):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.inference = inference
        self.opts = opts

        #Setting the field strength specific parameters
        if opts.scanner_type == '3T':
            self.diffusion_dir = 'Diffusion'
            self.shell_number = 4
            self.data_file = 'normalised_data.nii.gz'
            self.spatial_resolution = [len(subject_list),145,174,145]
        elif opts.scanner_type == '7T':
            self.diffusion_dir = 'Diffusion_7T'
            self.shell_number = 3
            self.data_file = 'data.nii.gz'
            self.spatial_resolution = [len(subject_list),173,207,173]
            
        
        pad_tens = (0,0,5,5,5,5,5,5)
        
        #Creating the dummy variables for the data to be loaded into RAM, the spatial resolution of the images changes depending on whether they are 3T or 7T:

        self.data_tensor = F.pad(torch.zeros(self.spatial_resolution+[opts.dwi_number]),pad_tens, mode = 'constant')
        self.gt_tensor = F.pad(torch.zeros(self.spatial_resolution+[47]),pad_tens, mode = 'constant')
        self.gt_fixel_tensor = F.pad(torch.zeros(self.spatial_resolution), (5,5,5,5,5,5), mode = 'constant')
        
        
        self.AQ_tensor = torch.zeros((len(subject_list),opts.dwi_number,47))
        self.gt_AQ_tensor = torch.zeros((len(subject_list),288,47))
        self.ttgen_mask_tensor = F.pad(torch.zeros(self.spatial_resolution+[5]),pad=pad_tens, mode = 'constant')
        self.wb_mask_tensor = F.pad(torch.zeros(self.spatial_resolution),(5,5,5,5,5,5), mode = 'constant')
    
        #Loading the data into the data tensor
        print('Loading the signal data into RAM')
        for i, subject in enumerate(subject_list):

            #Loading the undersampled signal into RAM
            path = os.path.join(self.data_dir, subject, 'T1w', self.diffusion_dir, opts.dwi_folder_name, self.data_file)
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj)),pad_tens, mode = 'constant')

        if self.inference:
            self.aff = nifti.affine
            self.head = nifti.header
        
        print(f'The shape of the data tensor is {self.data_tensor.shape}')

        print('Loading the ground truth fixel data into RAM')
        
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir, subject, 'T1w', self.diffusion_dir, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz')
            nifti = nib.load(path)
            self.gt_fixel_tensor[i,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj).astype(np.uint8)[:,:,:,0]),(5,5,5,5,5,5), mode = 'constant')


        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            #Should move the ground truth FOD to outside the undersampled folder to avoid this problem (note that it needs to be the who mrcat gt rather than just wm FOD)
            path = os.path.join(self.data_dir, subject,'T1w',self.diffusion_dir,'undersampled_fod','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj)),pad_tens, mode = 'constant')
        print(f'The shape of the ground truth tensor is {self.gt_tensor.shape}')

        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            #Importing the whole brain mask
            path_wb = os.path.join(self.data_dir,subject,'T1w',self.diffusion_dir,'nodif_brain_mask.nii.gz')
            nifti_wb = nib.load(path_wb)
            self.wb_mask_tensor[i,:,:,:] = F.pad(torch.tensor(np.array(nifti_wb.dataobj)),(5,5,5,5,5,5), mode = 'constant')
            
            #Importing the 5ttgen mask
            path_5ttgen = os.path.join(self.data_dir,subject,'T1w','5ttgen.nii.gz')
            nifti_5ttgen = nib.load(path_5ttgen)
            self.ttgen_mask_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti_5ttgen.dataobj))[:,:,:,:],pad_tens, mode = 'constant')
            
        #print(f'The shape of the mask tensor is {self.mask_tensor.shape}')
        

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting the undersampled b-vectors and b-values:
            bvecs = util.bvec_extract(self.data_dir, subject, self.diffusion_dir, opts.dwi_folder_name)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            bvals = util.bval_extract(self.data_dir, subject, self.diffusion_dir, opts.dwi_folder_name)
            
            #Extracting the ground truth b-vectors and b-values:
            gt_bvecs = util.bvec_extract(self.data_dir, subject, self.diffusion_dir)
            gt_bvecs_sph = util.ss_sph_coords(gt_bvecs)
            gt_bvecs_sph[gt_bvecs_sph[:,0]<0,0] = gt_bvecs_sph[gt_bvecs_sph[:,0]<0,0]+2*math.pi
            gt_bvals = util.bval_extract(self.data_dir, subject, self.diffusion_dir)
            
            
            #White matter response function extraction:
            order = 8
            with open(os.path.join(self.data_dir, subject,'T1w',self.diffusion_dir,opts.dwi_folder_name,'wm_response.txt'), 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(self.shell_number,6)
            for j in range(self.shell_number):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open(os.path.join(self.data_dir, subject,'T1w',self.diffusion_dir,opts.dwi_folder_name,'gm_response.txt'), 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open(os.path.join(self.data_dir, subject, 'T1w', self.diffusion_dir, opts.dwi_folder_name, 'csf_response.txt'), 'r') as txt:
                x = txt.read()
            g_csf = [float(resp) for resp in x.split('\n')[2:-1]]    
        
            
            self.AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals)
            self.gt_AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(gt_bvecs_sph, order, g_wm, g_gm, g_csf, gt_bvals)
        
        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

            #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        if self.opts.inference == True:
            self.coords = grid[self.wb_mask_tensor.to(bool),:]
        else:
            self.coords = grid[(self.ttgen_mask_tensor[:,:,:,:,0].to(bool) | self.ttgen_mask_tensor[:,:,:,:,1].to(bool) | self.ttgen_mask_tensor[:,:,:,:,2].to(bool)) & self.wb_mask_tensor.to(bool),:]

        #Printing the number of datapoints in the dataset:
        print('The number of datapoints in this dataset are: ' + str(len(self)))

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
       
        gt_fixel = self.gt_fixel_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3]]
        
        AQ = self.AQ_tensor[central_coords[0],:,:]
        gt_AQ = self.gt_AQ_tensor[central_coords[0],:,:]
        
        if self.inference:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), central_coords
        else:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), gt_AQ, gt_fixel.float()



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
            path = os.path.join(self.data_dir, subject,'T1w','Diffusion','undersampled_fod','undersampled_fod.nii.gz')
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))
        
        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir, subject,'T1w','Diffusion','undersampled_fod','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir,subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')

            nifti = nib.load(path)
            self.mask_tensor[i,:,:,:] = torch.tensor(np.array(nifti.dataobj))

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting bvectors from folders:
            bvecs = util.bvec_extract(self.data_dir, subject, 'undersampled_fod')
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            order = 8

            #Extracting bvalues:
            bvals = util.bval_extract(self.data_dir, subject, 'undersampled_fod')
            #White matter response function extraaction:
            with open(os.path.join(self.data_dir, subject,'T1w','Diffusion','undersampled_fod','wm_response.txt'), 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(4,6)
            for j in range(4):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open(os.path.join(self.data_dir, subject,'T1w','Diffusion','undersampled_fod','gm_response.txt'), 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open(os.path.join(self.data_dir, subject,'T1w','Diffusion','undersampled_fod','csf_response.txt'), 'r') as txt:
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
    def __init__(self, data_dir, subject_list, inference, opts,fixels):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.inference = inference
        self.fixels = fixels
        
        
        #Ommit if possible
        if opts.scanner_type == '3T':
            self.diffusion_dir = 'Diffusion'
            self.shell_number = 4
            self.data_file = 'normalised_data.nii.gz'
            self.spatial_resolution = [len(subject_list),145,174,145]
        elif opts.scanner_type == '7T':
            self.diffusion_dir = 'Diffusion_7T'
            self.shell_number = 3
            self.data_file = 'data.nii.gz'
            self.spatial_resolution = [len(subject_list),173,207,173]


        #Creating the dummy variables for the data to be loaded into RAM:
        self.data_tensor = torch.zeros((len(subject_list),79,87,97,opts.dwi_number))
        self.gt_tensor = torch.zeros((len(subject_list),79,87,97,47))
        self.gt_fixel_tensor = torch.zeros((len(subject_list),79,87,97))
        
        self.AQ_tensor = torch.zeros((len(subject_list),opts.dwi_number,47))
        self.mask_tensor = torch.zeros((len(subject_list),79,87,97))


        #Loading the data into the data tensor
        print('Loading the signal data into RAM')
        for i, subject in enumerate(subject_list):
            #path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','normalised_data.nii.gz')
            path = os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', opts.dwi_folder_name, 'normalised_data.nii.gz')
            nifti = nib.load(path)
            #Shape = [62, 70, 80, :]
            self.data_tensor[i,4:66,4:74,4:84,:] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124,:]

        print('Loading the ground truth fixel data into RAM')
        
        for i, subject in enumerate(subject_list):
            path = os.path.join(self.data_dir, subject, 'T1w', self.diffusion_dir, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz')
            nifti = nib.load(path)
            self.gt_fixel_tensor[i,4:66,4:74,4:84] = torch.tensor(np.array(nifti.dataobj).astype(np.uint8)[13:75, 90:160, 44:124,0])

        #Loading the ground truth data into RAM
        print('Loading the ground Truth FOD data into RAM')
        for i, subject in enumerate(subject_list):
            #path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','undersampled_fod','gt_fod.nii.gz')
            path = os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', opts.dwi_folder_name, 'gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,4:66,4:74,4:84,:] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124,:]
        self.aff = nifti.affine

        print('Loading the ground truth fixel data into RAM')
        if self.fixels:
            for i, subject in enumerate(subject_list):
                path = os.path.join(self.data_dir, subject, 'T1w','Diffusion' , 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz')
                nifti = nib.load(path)
                self.gt_fixel_tensor[i,4:66,4:74,4:84] = torch.tensor(np.array(nifti.dataobj).astype(np.uint8)[13:75, 90:160, 44:124,0])


        #Loading the mask data into RAM
        print('Loading the mask data into RAM')
        for i, subject in enumerate(subject_list):
            #path = os.path.join('/media','duanj','F','joe','hcp_2',subject,'T1w','Diffusion','nodif_brain_mask.nii.gz')
            path = os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', 'nodif_brain_mask.nii.gz')
            nifti = nib.load(path)
            self.mask_tensor[i,4:66,4:74,4:84] = torch.tensor(np.array(nifti.dataobj))[13:75, 90:160, 44:124]

        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')
        for i, subject in enumerate(subject_list):
            #Extracting bvectors from folders:
            bvecs = util.bvec_extract(self.data_dir, subject, self.diffusion_dir, opts.dwi_folder_name)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            order = 8

            #Extracting bvalues:
            bvals = util.bval_extract(self.data_dir, subject, self.diffusion_dir, opts.dwi_folder_name)

            #White matter response function extraaction:
            with open(os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', opts.dwi_folder_name, 'wm_response.txt'), 'r') as txt:
                
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(4,6)
            for j in range(4):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open(os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', opts.dwi_folder_name, 'gm_response.txt'), 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open(os.path.join(self.data_dir, subject, 'T1w', 'Diffusion', opts.dwi_folder_name, 'csf_response.txt'), 'r') as txt:
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
        
        #Printing the length of the experimental dataset:
        print('The length of the experimental dataset is: ' + str(len(self)))
    

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
        gt_fixel = self.gt_fixel_tensor[central_coords[0], central_coords[1], central_coords[2], central_coords[3]]

        AQ = self.AQ_tensor[central_coords[0],:,:]
        if self.inference:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), central_coords
        else:
            return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), gt_fixel.float()

class UndersampleDataset(torch.utils.data.Dataset):
    def __init__(self, subject, data_path, normalised = False, sample_pattern = 'uniform', undersample_val = 6, T7 = False, save_folder = 'undersampled_fod'):
        
        #Initialising the parameters for the dataset class.
        self.normalised = normalised
        self.subject = subject
        self.img_dir = data_path
        self.sample_pattern = sample_pattern
        self.data_path = data_path
        self.undersample_val = undersample_val
        self.T7 = T7
        self.save_folder=save_folder

        #Setting the field strength specific parameters
        if T7 == False:
            self.diffusion_dir = 'Diffusion'
            self.shell_number = 4
            self.data_file = 'normalised_data.nii.gz'
        elif T7 == True:
            self.diffusion_dir = 'Diffusion_7T'
            self.shell_number = 3
            self.data_file = 'data.nii.gz'

        #Calculating the mask list and keep lists (using this function here will only work when a constant undersampling pattern is used)
        self.mask_list, self.keep_list= self.sample_lists()

        #Creating the data path and the mask path.
        if self.T7 == True:
            dwi_path = os.path.join(data_path, self.subject, 'T1w', 'Diffusion_7T', 'data.nii.gz')
        else:
            dwi_path = os.path.join(data_path, self.subject, 'T1w', 'Diffusion', 'data.nii.gz')
        
        #Loading the data for the subject
        image = nib.load(dwi_path)
        self.head = image.header
        self.aff = image.affine
        print('Loading image data')
        self.image = torch.tensor(image.get_fdata())
        
        #os.mkdir(path = data_path+'/'+subject+'/T1w/Diffusion/undersampled_fod_resptest_1')

    

    def __len__(self):
        return int(torch.sum(torch.tensor(self.mask)))
    

    def data_save(self):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        print('Saving data')
        im_usamp = nib.Nifti1Image(self.image[:,:,:,self.keep_list].float().detach().numpy(), affine=self.aff)
        if self.T7 == True:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion_7T', self.save_folder, 'data.nii.gz')
        else:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', self.save_folder, 'data.nii.gz')
        #save_path = '/media/duanj/F/joe/Project_1_recon/FODNet/dataset/104820/LARDI_data/data_b1000_g32.nii.gz'
        nib.save(im_usamp, save_path)
        print('Finished saving data')

    def all_save(self):
        self.data_save()
        self.bval_save()
        self.bvec_save()
        #self.bval_normalised_save()

    def sample_lists(self):
        '''
        A function which returns two lists - the mask_list, which is the q-space volumes to be masked,
        and keep_list for the q-space volumes not to be masked. This will be used in the __getitem__ function 
        of this class.
        '''

        self.bvals = self.bval_extract()
        if self.T7 == True:
            b0_list = []
            b1000_list = []
            b2000_list = []

            for i in range(len(self.bvals)):
                if self.bvals[i] <100:
                    b0_list.append(i)
                elif 960<self.bvals[i]<1040:
                    b1000_list.append(i)
                elif 1960<self.bvals[i]<2040:
                    b2000_list.append(i)

            mask_list = []
            keep_list = []

            for i in range(len(b1000_list)):
                    if i>=self.undersample_val:
                        mask_list.append(i)
                    else:
                        keep_list.append(i)
            
            keep_list = torch.tensor([b1000_list[i]for i in keep_list] +
                                    [b2000_list[i]for i in keep_list] +
                                    b0_list[:4])


        
        else:
            b0_list = []
            b1000_list = []
            b2000_list = []
            b3000_list = []
        
            for i in range(len(self.bvals)):
                if self.bvals[i] <20:
                    b0_list.append(i)
                elif 980<self.bvals[i]<1020:
                    b1000_list.append(i)
                elif 1980<self.bvals[i]<2020:
                    b2000_list.append(i)
                elif 2980<self.bvals[i]<3020:
                    b3000_list.append(i)

            #per_shell_undersample = int(torch.floor(torch.tensor(self.undersample_num)/3))
            #b1000_mask_list = random.sample(b1000_list,per_shell_undersample)
            #b2000_mask_list = random.sample(b2000_list,per_shell_undersample)
            #b3000_mask_list = random.sample(b3000_list,per_shell_undersample)

            mask_list = []
            keep_list = []
            #Just a test for undersampling - need to construct a better way of undersampling.
            for i in range(len(b1000_list)):
                    if i >= self.undersample_val:
                        mask_list.append(i)
                    else:
                        keep_list.append(i)
            
            keep_list = [i for i in range(9,18)]
            
            print(keep_list)
            mask_list = torch.tensor([b1000_list[i] for i in mask_list])

            #Uncomment this for the multi-shell version 
            keep_list = torch.tensor([b1000_list[i]for i in keep_list] +
                                    [b2000_list[i]for i in keep_list] +
                                    [b3000_list[i]for i in keep_list] +
                                    b0_list[:3])

            mask_list = torch.tensor([])
    
        return mask_list, keep_list
    
    def bval_extract(self):
        '''
        Input:
            subj (str) - The subject whose bvals you want to extract.
            path (str) - The path to where the data is located (subjects location).
        
        Desc:
            A function to extract the bvalues from the file they are located in and to return them as a list of values.
        '''
        #print('Saving bvals')
        if self.T7 == True:
            path = os.path.join(self.data_path, self.subject, 'T1w', 'Diffusion_7T', 'bvals') 
        else:    
            path = os.path.join(self.data_path, self.subject, 'T1w', 'Diffusion', 'bvals') 
        
        bvals = open(path, 'r')
        
        bvals_str = bvals.read()
        bvals = [int(b) for b in bvals_str.split()]
        #print('Finished saving bvals')
        return bvals

    def bvec_save(self):
        print('Saving bvectors')
        #Undersampled bvector calculation.
        if self.T7 == True:
            path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion_7T', 'bvecs')
        else:
            path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', 'bvecs')
        
        with open(path ,'r') as temp:
            bvecs = temp.read()

        bvecsxyz = bvecs.split('\n')
        bvecsxyz.pop(3)

        xvals = [x for x in bvecsxyz[0].split()]
        yvals = [y for y in bvecsxyz[1].split()]
        zvals = [z for z in bvecsxyz[2].split()]

        #Undersampled bvectors
        xvals_new = [xvals[ind] for ind in self.keep_list]
        yvals_new = [yvals[ind] for ind in self.keep_list]
        zvals_new = [zvals[ind] for ind in self.keep_list]

        xvals_str = ' '.join(xvals_new)
        yvals_str = ' '.join(yvals_new)
        zvals_str = ' '.join(zvals_new)

        bvecs_string = '\n'.join((xvals_str,yvals_str,zvals_str))
        #save_path = '/media/duanj/F/joe/Project_1_recon/FODNet/dataset/104820/LARDI_data/data_b1000_g32_bvecs'
        if self.T7 == True:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion_7T', self.save_folder, 'bvecs')
        else:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', self.save_folder, 'bvecs')
        
        with open(save_path, 'w') as temp:
            temp.write(bvecs_string)
        print('Finished Saving bvectors')

    def bvec_target_save(self):
        '''
        A function save the text file containing the bvectors which belong to the DWI images we are reconstructing. This is the bvectors which correspond to the masked image indexes. 
        This function will be particularly useful for implementing models which are conditioned on the bvector to generate a certain direction such as https://github.com/m-lyon/dMRI-RCNN.
        '''
        path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', 'bvecs')
        with open(path ,'r') as temp:
            bvecs = temp.read()

        bvecsxyz = bvecs.split('\n')
        bvecsxyz.pop(3)

        xvals = [x for x in bvecsxyz[0].split()]
        yvals = [y for y in bvecsxyz[1].split()]
        zvals = [z for z in bvecsxyz[2].split()]

        #Undersampled bvectors
        xvals_tgt = [xvals[ind] for ind in self.mask_list]
        yvals_tgt = [yvals[ind] for ind in self.mask_list]
        zvals_tgt = [zvals[ind] for ind in self.mask_list]

        xvals_str = ' '.join(xvals_tgt)
        yvals_str = ' '.join(yvals_tgt)
        zvals_str = ' '.join(zvals_tgt)

        bvecs_string = '\n'.join((xvals_str,yvals_str,zvals_str))

        save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', self.save_folder, 'bvecs')
        #save_path = '/media/duanj/F/joe/Project_1_recon/FODNet/dataset/104820/LARDI_data/data_b1000_g32_bvecs'
        with open(save_path, 'w') as temp:
            temp.write(bvecs_string)
    
    def bval_save(self):
        print('Saving bvals')
        ##Bvalues
        new_bvals = [str(self.bvals[ind]) for ind in self.keep_list]
        bvals_string = ' '.join(new_bvals)

        if self.T7 == True:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion_7T', self.save_folder, 'bvals')
        else:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', self.save_folder, 'bvals')
        #save_path = '/media/duanj/F/joe/Project_1_recon/FODNet/dataset/104820/LARDI_data/data_b1000_g32.bvals'
        with open(save_path,'w') as temp:
            temp.write(bvals_string)
        print('Finished saving bvals')

    def bval_normalised_save(self):
        print('Saving bvals')
        ##Bvalues
        new_bvals = [str(self.bvals[ind]) for ind in self.keep_list]
        new_bvals[-4:] = [0,0,0,0] 
        bvals_string = ' '.join(new_bvals)

        if self.T7 == True:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion_7T', self.save_folder, 'bvals_normalised')
        else:
            save_path = os.path.join(self.img_dir, self.subject, 'T1w', 'Diffusion', self.save_folder, 'bvals_normalised')
        #save_path = '/media/duanj/F/joe/Project_1_recon/FODNet/dataset/104820/LARDI_data/data_b1000_g32.bvals'
        with open(save_path,'w') as temp:
            temp.write(bvals_string)
        print('Finished saving bvals')

def init_dataloaders(opts):
    #Write a function in data.py to initialise the dataset and dataloader. - Clean up this part of the code.
    if opts.dataset_type == 'all':
        d_train = DWIPatchDataset(opts.data_dir, opts.train_subject_list, inference=False, opts=opts)
        d_val = DWIPatchDataset(opts.data_dir, opts.val_subject_list, inference=False, opts=opts)
    elif opts.dataset_type == 'experiment':
        d_train = ExperimentPatchDataset(opts.data_dir, ['100206'], inference=False, opts=opts)
        d_val = ExperimentPatchDataset(opts.data_dir, ['100307'], inference=False, opts=opts)

    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=opts.batch_size,
                                            shuffle=True, num_workers=opts.train_workers, 
                                            drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=2,
                                            shuffle=True, num_workers=opts.val_workers,
                                            drop_last = True)
    
    return train_dataloader, val_dataloader