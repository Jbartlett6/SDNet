import torch
import sys
import os
sys.path.append(os.path.join(sys.path[0],'..'))
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys 
import argparse
import data
import argparse

class UndersampleDataset(torch.utils.data.Dataset):
    def __init__(self, subject, data_path, normalised = False, sample_pattern = 'uniform', undersample_val = 9, T7 = False, save_folder = 'undersampled_fod'):
        
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
        
        self.diffusion_dir = 'Diffusion'
        self.shell_number = 4
        self.data_file = 'normalised_data.nii.gz'

        #Calculating the mask list and keep lists (using this function here will only work when a constant undersampling pattern is used)
        self.mask_list, self.keep_list= self.sample_lists()

        #Creating the data path and the mask path.
       
       
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

            mask_ind = []
            keep_ind = []
            #Just a test for undersampling - need to construct a better way of undersampling.
            for i in range(len(b1000_list)):
                    if i >= self.undersample_val:
                        mask_ind.append(i)
                    else:
                        keep_ind.append(i)
            
            
            
            print(keep_list)
            #mask_list = torch.tensor([b1000_list[i] for i in mask_list])

            #Uncomment this for the multi-shell version 
            keep_list = torch.tensor([b1000_list[i]for i in keep_ind] +
                                    [b2000_list[i]for i in keep_ind] +
                                    [b3000_list[i]for i in keep_ind] +
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

print('Running dwi_undersampled.py:')



################################################Parameters for the undersamplding process ############################################
subject_list = ['174437',
'318637',
'581450',
'145127',
'147737',
'178849',
'130821']
data_path = '/media/duanj/F/joe/hcp_2'
save_folder = 'undersampled_fod'
dwi_per_shell = 9
bool_7T = False
################################################Parameters for the undersampleing process ############################################

#usamp = int(sys.argv[1])
#usamp = args.undersampling_rate
for subject in subject_list:
#os.mkdir(os.path.join(args.data_path, args.subject,'T1w','Diffusion_7T',args.save_folder))
    print('Initialising dataset')
    print(subject)
    d = UndersampleDataset(subject, data_path , undersample_val=dwi_per_shell, T7=bool_7T, save_folder=save_folder)
    print('Dataset initialised')
    d.all_save()


#Normalising the data:
for subject in subject_list:
    subject_undersampled_path = os.path.join(data_path,subject,'T1w','Diffusion',save_folder)
    usamp_data_path = os.path.join(subject_undersampled_path,'data.nii.gz')
    norm_usamp_data_path = os.path.join(subject_undersampled_path, 'normalised_data.nii.gz')
    usamp_bvecs_path = os.path.join(subject_undersampled_path, 'bvecs')
    usamp_bvals_path = os.path.join(subject_undersampled_path, 'bvals')

    #Paths relating to the response functions:
    wm_response = os.path.join(subject_undersampled_path, 'wm_response.txt')
    gm_response = os.path.join(subject_undersampled_path, 'gm_response.txt')
    csf_response = os.path.join(subject_undersampled_path, 'csf_response.txt')
    
    #Normalising the data and calculating the normalised response functions using Mrtrix.
    os.system('dwinormalise individual ' + str(usamp_data_path) + ' ' + str(norm_usamp_data_path) + ' -fslgrad ' + str(usamp_bvecs_path) + ' ' + str(usamp_bvals_path) + ' -intensity 1')
    os.system('dwi2response dhollander ' + str(norm_usamp_data_path) + ' ' + str(wm_response) + ' ' + str(gm_response) + ' ' +  str(csf_response) + ' -fslgrad '+str(usamp_bvecs_path) + ' ' + str(usamp_bvals_path))
    


