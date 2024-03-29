import utils.util as util

import math
import os 

import torch 
import torch.nn.functional as F
import nibabel as nib 
import numpy as np


class DWIPatchDataset(torch.utils.data.Dataset):
    '''
    Description:    
                - A dataset class for the purpose of FOD reconstruction. This class
                is used to load the data into RAM.  The data must be in the same format as 
                outlined in the repository. 
    Methods:
                __init__                    - Initialising the parameters of the dataset. Some of this method 
                                            involves setting the options which will be used later in the data-set, 
                                            such as data paths and sizes, majority of work is done in init_data_tensors
                                            which loads the data into RAM.
                __len__                     - The number of coords in self.coords. 
                __getitem__                 - Retrieves item by using co-ords to convert idx to indices. 
                init_data_tensors           - Method to load the data into RAM
                load_input_signal           - Loads the input DWI signals into RAM
                load_gt_fixel               - Loads the target fiel values (calculated in MRtrix3) into RAM
                load_gt_fod                 - Loads the ground truth target FODs into RAM
                load_brain_masks            - Loads the 5ttgen and whole brain masks into RAM
                load_convolution_matricies  - Loads the convolution matrices - these componenets are needed for DWI cosnsitency layers.
                load_coords                 - The co-ordinates (with respect to the whole dataset) of voxels which will be loaded by 
                                            the dataset. Depending on whether the dataset is being used for training or inference these 
                                            will be the co-ordinates of voxels in wm and gm only or the whole brain. 

    '''
    def __init__(self, subject_list, inference, training_voxels, opts):
        
        #Initialising the parameters for the dataset class.
        self.subject_list = subject_list
        self.inference = inference
        self.training_voxels = training_voxels
        self.opts = opts
        
        # Extracting the dimension of the images using a sample image.
        sample_path = os.path.join(self.opts.data_dir, subject_list[0], 'T1w', 'T1w_acpc_dc_restore_1.25.nii.gz')
        nifti = nib.load(sample_path)
        image_dims = nifti.shape
        self.spatial_resolution = [len(subject_list)] + list(image_dims)

        #Setting the padding tensor for the image - key if the brain mask is less than 9 voxels away from the edge of the image.
        self.pad_tens = (0,0,5,5,5,5,5,5)
        
        #Initialising the tensors which the data will be stored in.
        self.init_data_tensors()
        
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
        
        return input_signals.float().unsqueeze(-1), target_fod.float(), AQ.float(), gt_fixel.float(), central_coords
    
    def init_data_tensors(self):
        self.load_input_signal()
        self.load_gt_fixel()
        self.load_gt_fod()
        self.load_brain_masks()
        self.load_convolution_matricies()
        self.load_coords()
        
    def load_input_signal(self):
        #Loading the DWI signal data into the data tensor
        print('Loading the signal data into RAM')

        #Defining the input signal tensor.
        self.data_tensor = F.pad(torch.zeros(self.spatial_resolution+[self.opts.dwi_number]),self.pad_tens, mode = 'constant')

        for i, subject in enumerate(self.subject_list):

            path = os.path.join(self.opts.data_dir, subject, 'T1w', self.opts.diffusion_dir, self.opts.dwi_folder_name, self.opts.data_file)
            nifti = nib.load(path)
            self.data_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj)),self.pad_tens, mode = 'constant')

        #If performing inference need the affine and header information to allow the nifti file to be saved.
        if self.inference:
            self.aff = nifti.affine
            self.head = nifti.header
        
        print(f'The shape of the data tensor is {self.data_tensor.shape}')
    
    def load_gt_fixel(self):
        #Loading the ground truth fixel data into the gt_fixel_tensor tensor.
        print('Loading the ground truth fixel data into RAM')

        #Defining the ground truth fixel tensor
        self.gt_fixel_tensor = F.pad(torch.zeros(self.spatial_resolution), (5,5,5,5,5,5), mode = 'constant')

        for i, subject in enumerate(self.subject_list):
            path = os.path.join(self.opts.data_dir, subject, 'T1w', self.opts.diffusion_dir, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz')
            nifti = nib.load(path)
            self.gt_fixel_tensor[i,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj).astype(np.uint8)[:,:,:,0]),(5,5,5,5,5,5), mode = 'constant')

    def load_gt_fod(self):
        #Loading the ground truth 
        print('Loading the ground Truth FOD data into RAM')

        #Defining the ground truth FOD tensor
        self.gt_tensor = F.pad(torch.zeros(self.spatial_resolution+[47]),self.pad_tens, mode = 'constant')

        for i, subject in enumerate(self.subject_list):
            #Should move the ground truth FOD to outside the undersampled folder to avoid this problem (note that it needs to be the who mrcat gt rather than just wm FOD)
            path = os.path.join(self.opts.data_dir, subject,'T1w','Diffusion','gt_fod.nii.gz')
            nifti = nib.load(path)
            self.gt_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti.dataobj)),self.pad_tens, mode = 'constant')
        print(f'The shape of the ground truth tensor is {self.gt_tensor.shape}')

    def load_brain_masks(self):
        #Loading the mask data into RAM
        print('Loading the mask data into RAM')

        #Defining the 5ttgen mask and whole brain mask tensors.
        self.ttgen_mask_tensor = F.pad(torch.zeros(self.spatial_resolution+[5]),pad=self.pad_tens, mode = 'constant')
        self.wb_mask_tensor = F.pad(torch.zeros(self.spatial_resolution),(5,5,5,5,5,5), mode = 'constant')

        for i, subject in enumerate(self.subject_list):
            #Importing the whole brain mask
            path_wb = os.path.join(self.opts.data_dir,subject,'T1w',self.opts.diffusion_dir,'nodif_brain_mask.nii.gz')
            nifti_wb = nib.load(path_wb)
            self.wb_mask_tensor[i,:,:,:] = F.pad(torch.tensor(np.array(nifti_wb.dataobj)),(5,5,5,5,5,5), mode = 'constant')
            
            #Importing the 5ttgen mask
            path_5ttgen = os.path.join(self.opts.data_dir,subject,'T1w','5ttgen.nii.gz')
            nifti_5ttgen = nib.load(path_5ttgen)
            self.ttgen_mask_tensor[i,:,:,:,:] = F.pad(torch.tensor(np.array(nifti_5ttgen.dataobj))[:,:,:,:],self.pad_tens, mode = 'constant')
       
    def load_convolution_matricies(self):
        
        #Loading the spherical harmonic co-ords into RAM.
        print('Loading the Spherical Convolution co-ords into RAM')

        #Defining the AQ tensor
        self.AQ_tensor = torch.zeros((len(self.subject_list),self.opts.dwi_number,47))
        
        for i, subject in enumerate(self.subject_list):
            #Extracting the undersampled b-vectors and b-values:
            bvecs = util.bvec_extract(self.opts.data_dir, subject, self.opts.diffusion_dir, self.opts.dwi_folder_name)
            bvecs_sph = util.ss_sph_coords(bvecs)
            bvecs_sph[bvecs_sph[:,0]<0,0] = bvecs_sph[bvecs_sph[:,0]<0,0]+2*math.pi
            bvals = util.bval_extract(self.opts.data_dir, subject, self.opts.diffusion_dir, self.opts.dwi_folder_name)
            
            #Extracting the ground truth b-vectors and b-values:
            gt_bvecs = util.bvec_extract(self.opts.data_dir, subject, self.opts.diffusion_dir)
            gt_bvecs_sph = util.ss_sph_coords(gt_bvecs)
            gt_bvecs_sph[gt_bvecs_sph[:,0]<0,0] = gt_bvecs_sph[gt_bvecs_sph[:,0]<0,0]+2*math.pi
            gt_bvals = util.bval_extract(self.opts.data_dir, subject, self.opts.diffusion_dir)
            
        
            #White matter response function extraction:
            order = 8
            with open(os.path.join(self.opts.data_dir, subject,'T1w',self.opts.diffusion_dir,self.opts.dwi_folder_name,'wm_response.txt'), 'r') as txt:
                x = txt.read()
            x = x.split('\n')[2:-1]
            
            g_wm = torch.zeros(self.opts.shell_number,6)
            for j in range(self.opts.shell_number):
                g_wm[j] = torch.tensor([float(resp) for resp in x[j].split(' ')])

            #Grey matter response function extraction:
            with open(os.path.join(self.opts.data_dir, subject,'T1w',self.opts.diffusion_dir,self.opts.dwi_folder_name,'gm_response.txt'), 'r') as txt:
                x = txt.read()
            g_gm = [float(resp) for resp in x.split('\n')[2:-1]]

            #CSF response function extraction:
            with open(os.path.join(self.opts.data_dir, subject, 'T1w', self.opts.diffusion_dir, self.opts.dwi_folder_name, 'csf_response.txt'), 'r') as txt:
                x = txt.read()
            g_csf = [float(resp) for resp in x.split('\n')[2:-1]]    
        
            
            self.AQ_tensor[i,:,:] = util.construct_sh_basis_msmt_all(bvecs_sph, order, g_wm, g_gm, g_csf, bvals)

    def load_coords(self):
        print('Creating Co-ordinate grid')
        #Creating a meshgrid for the subject - a volume where each voxel is the x,y,z coordinate.
        seq_0 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[0])])
        seq_1 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[1])])
        seq_2 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[2])])
        seq_3 = torch.tensor([i for i in range(self.wb_mask_tensor.shape[3])])

        grid_0, grid_1, grid_2, grid_3 = torch.meshgrid(seq_0, seq_1, seq_2, seq_3)
        grid = torch.stack((grid_0, grid_1, grid_2, grid_3), 4)

        #Making a vector containing the co-ordinates of only pixels which are in the brain mask.
        if self.training_voxels == False:
            self.coords = grid[self.wb_mask_tensor.to(bool),:]
        else:
            self.coords = grid[(self.ttgen_mask_tensor[:,:,:,:,0].to(bool) | self.ttgen_mask_tensor[:,:,:,:,1].to(bool) | self.ttgen_mask_tensor[:,:,:,:,2].to(bool)) & self.wb_mask_tensor.to(bool),:]


def init_dataloaders(opts):
    #Write a function in data.py to initialise the dataset and dataloader. - Clean up this part of the code.
    d_train = DWIPatchDataset(opts.train_subject_list, training_voxels = True, inference=False, opts=opts)
    d_val = DWIPatchDataset(opts.val_subject_list, training_voxels = True, inference=False, opts=opts)

    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=opts.batch_size,
                                            shuffle=True, num_workers=opts.train_workers, 
                                            drop_last = True)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=256,
                                            shuffle=True, num_workers=opts.val_workers,
                                            drop_last = True)
    
    return train_dataloader, val_dataloader