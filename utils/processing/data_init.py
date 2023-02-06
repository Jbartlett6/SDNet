import os
import torch
import nibabel as nib
import numpy as np
import shutil
'''
This script aims to gather all of the preprocessing and data creation into one place. 
The majority of thsi script is a class which stores and manipulates file paths so that the files can easily 
be manipulated/updated from python rather than having to use bash scripts
'''

class data_manager():
    def __init__(self):
        self.data_dir = '/media/duanj/E44441FB4441D0CA/jxb/Projects/data/hcp'
        # self.subject_list = ['100206',
        #                     '100307',
        #                     '100408',
        #                     '100610',
        #                     '101006',
        #                     '101107',
        #                     '101309',
        #                     '101410',
        #                     '101915',
        #                     '102008',
        #                     '102109',
        #                     '102311',
        #                     '102513',
        #                     '102614',
        #                     '102715',
        #                     '102816',
        #                     '103010',
        #                     '103111',
        #                     '103212',
        #                     '103414',
        #                     '103515',
        #                     '103818',
        #                     '104012',
        #                     '104416',
        #                     '104820',
        #                     '174437',
        #                     '318637',
        #                     '581450',
        #                     '145127',
        #                     '147737',
        #                     '178849',
        #                     '130821']    
        self.subject_list = ['103818']     

    def set_subject_paths(self, subject):
        '''
        Sets the paths sepcific to the input subject.
        '''
        #Setting the class subject attribute
        self.current_subject=subject

        #Setting all of the paths:
        self.set_T1w_paths()
        self.set_diffusion_paths()
        self.set_fixel_directory_paths()
        self.set_tractseg_paths()
        self.set_undersampled_fod_dir()

    def set_T1w_paths(self): 
        #T1w paths:
        self.T1w_dir = os.path.join(self.data_dir, self.current_subject, 'T1w')
        self.T1w_image_path = os.path.join(self.T1w_dir, 'T1w_acpc_dc_restore_1.25.nii.gz')
        self.fivettgen_path = os.path.join(self.T1w_dir, '5ttgen.nii.gz')
        self.wm_mask_path = os.path.join(self.T1w_dir, 'white_matter_mask.nii.gz')

    def set_diffusion_paths(self):
        #Diffusion Paths:
        self.diffusion_dir = os.path.join(self.T1w_dir, 'Diffusion')
        self.gt_bvals_path = os.path.join(self.diffusion_dir, 'bvals')
        self.gt_bvecs_path = os.path.join(self.diffusion_dir, 'bvecs')
        self.gt_data_path = os.path.join(self.diffusion_dir, 'data.nii.gz')
        self.brain_mask_path = os.path.join(self.diffusion_dir, 'nodif_brain_mask.nii.gz')
        self.gt_normalised_data_path = os.path.join(self.diffusion_dir, 'normalised_data.nii.gz')

        self.gt_wm_response_path = os.path.join(self.diffusion_dir, 'wm_response.txt')
        self.gt_gm_response_path = os.path.join(self.diffusion_dir, 'gm_response.txt')
        self.gt_csf_response_path = os.path.join(self.diffusion_dir, 'csf_response.txt')

        self.gt_wm_fod_path = os.path.join(self.diffusion_dir, 'wmfod.nii.gz')
        self.gt_gm_fod_path = os.path.join(self.diffusion_dir, 'gm.nii.gz')
        self.gt_csf_fod_path = os.path.join(self.diffusion_dir, 'csf.nii.gz')
        self.gt_whole_fod_path = os.path.join(self.diffusion_dir, 'gt_whole_fod.nii.gz')

        

    def set_fixel_directory_paths(self):
        #Fixel_directory
        self.fixel_directory = os.path.join(self.diffusion_dir, 'fixel_directory')
        self.index_mif = os.path.join(self.fixel_directory, 'index.mif')
        self.index_nifti = os.path.join(self.fixel_directory, 'index.nii.gz')
        self.index_channel_1_path = os.path.join(self.fixel_directory, 'index_1.nii.gz') 
        self.afd_path = os.path.join(self.fixel_directory, 'afd.nii.gz')
        self.afd_im_path = os.path.join(self.fixel_directory, 'afd_im.nii.gz')
        self.peak_amp_path = os.path.join(self.fixel_directory, 'peak_amp.nii.gz')
        self.peak_amp_im_path = os.path.join(self.fixel_directory, 'peak_amp_im.nii.gz')
        self.fixel_directions_path = os.path.join(self.fixel_directory, 'directions.nii.gz')  
        self.thresholded_fixels_path = os.path.join(self.fixel_directory, 'thresholded_fixels.nii.gz')
    
    def set_tractseg_paths(self):
        #Tractseg paths:
        self.tractseg_dir = os.path.join(self.diffusion_dir, 'tractseg')
        self.bundle_segmentations_dir = os.path.join(self.tractseg_dir, 'bundle_segmentations')
        self.CC_path = os.path.join(self.bundle_segmentations_dir, 'CC.nii.gz')
        self.MCP_path = os.path.join(self.bundle_segmentations_dir, 'MCP.nii.gz')

        self.CST_left_path = os.path.join(self.bundle_segmentations_dir, 'CST_left.nii.gz')
        self.CST_right_path = os.path.join(self.bundle_segmentations_dir, 'CST_right.nii.gz')
        self.CST_whole_path = os.path.join(self.bundle_segmentations_dir, 'CST_whole.nii.gz')

        self.SLF_1_right_path = os.path.join(self.bundle_segmentations_dir, 'SLF_I_right.nii.gz')
        self.SLF_1_left_path = os.path.join(self.bundle_segmentations_dir, 'SLF_I_left.nii.gz')
        self.SLF_2_right_path = os.path.join(self.bundle_segmentations_dir, 'SLF_II_right.nii.gz')
        self.SLF_2_left_path = os.path.join(self.bundle_segmentations_dir, 'SLF_II_left.nii.gz')
        self.SLF_3_right_path = os.path.join(self.bundle_segmentations_dir, 'SLF_III_right.nii.gz')
        self.SLF_3_left_path = os.path.join(self.bundle_segmentations_dir, 'SLF_III_left.nii.gz')
        self.SLF_whole_path = os.path.join(self.bundle_segmentations_dir, 'SLF_whole.nii.gz')

        self.CC_1fixel_path = os.path.join(self.bundle_segmentations_dir, 'CC_1fixel.nii.gz')
        self.MCP_CST_2fixel_path = os.path.join(self.bundle_segmentations_dir, 'MCP_CST_2fixel.nii.gz')
        self.CC_CST_SLF_3fixel_path = os.path.join(self.bundle_segmentations_dir, 'CC_CST_SLF_3fixel.nii.gz')

    def set_undersampled_fod_dir(self):
        #Undersampled FOD directory
        self.undersampled_fod_dir = os.path.join(self.diffusion_dir, 'undersampled_fod')
        self.undersampled_bvals_path = os.path.join(self.diffusion_dir, 'bvals')
        self.undersampled_bvecs_path = os.path.join(self.diffusion_dir, 'bvecs')
        self.undersampled_data_path = os.path.join(self.diffusion_dir, 'data.nii.gz')
        self.undersampled_normalised_data_path = os.path.join(self.diffusion_dir, 'normalised_data.nii.gz')

        self.undersampled_wm_response_path = os.path.join(self.undersampled_fod_dir, 'wm_response.txt')
        self.undersampled_gm_response_path = os.path.join(self.undersampled_fod_dir, 'gm_response.txt')
        self.undersampled_csf_response_path = os.path.join(self.undersampled_fod_dir, 'csf_response.txt')

        self.undersampled_wm_fod_path = os.path.join(self.undersampled_fod_dir, 'wmfod.nii.gz')
        self.undersampled_gm_fod_path = os.path.join(self.undersampled_fod_dir, 'gm.nii.gz')
        self.undersampled_csf_fod_path = os.path.join(self.undersampled_fod_dir, 'csf.nii.gz')

    def normalise_gt_data(self):
        os.system('dwinormalise individual' + f" \'{self.gt_data_path}\'" + f" \'{self.brain_mask_path}\'" + f" \'{self.gt_normalised_data_path}\'" + " -fslgrad" + f" \'{self.gt_bvecs_path}\'" + f" \'{self.gt_bvals_path}\'" + " -intensity 1")
    
    def calc_gt_fod(self):
        os.system('dwi2response dhollander' + f" \'{self.gt_normalised_data_path}\'" + f" \'{self.gt_wm_response_path}\'" + f" \'{self.gt_gm_response_path}\'" + f" \'{self.gt_csf_response_path}\'" + " -fslgrad" + f" \'{self.gt_bvecs_path}\'" + f" \'{self.gt_bvals_path}\'")
        os.system('dwi2fod msmt_csd -fslgrad' + f" \'{self.gt_bvecs_path}\'" + f" \'{self.gt_bvals_path}\'" + f" \'{self.gt_normalised_data_path}\'" + f" \'{self.gt_wm_response_path}\'" + f" \'{self.gt_wm_fod_path}\'" + f" \'{self.gt_gm_response_path}\'" + f" \'{self.gt_gm_fod_path}\'" + f" \'{self.gt_csf_response_path}\'" + f" \'{self.gt_csf_fod_path}\'")
        os.system(f"mrcat -axis 3 {self.gt_wm_fod_path} {self.gt_gm_fod_path} {self.gt_csf_fod_path} {self.gt_whole_fod_path}")
    
    def fill_diffusion_dir(self):
        self.normalise_gt_data()
        self.calc_gt_fod()

    def create_thresholded_fixels(self):
        self.thresholded_fixels_path

        nifti = nib.load(self.index_nifti)
        aff = nifti.affine

        fix_im = np.array(nifti.dataobj)[:,:,:,0]
        fix_im[fix_im >= 4] = 4

        nifti_updated = nib.Nifti1Image(fix_im, aff)

        nib.save(nifti_updated, self.thresholded_fixels_path)
    
    def fill_gt_fixel_dir(self):
        os.system('fod2fixel -afd afd.nii.gz -peak_amp peak_amp.nii.gz' + f" \'{self.gt_wm_fod_path}\'" + f" \'{self.fixel_directory}\'")
        os.system("mrconvert" + f" \'{self.index_mif}\'" + f" \'{self.index_nifti}\'")
        if os.path.exists(self.index_mif):
            os.remove(self.index_mif)

        os.system('fixel2voxel -number 11' + f" \'{self.afd_path}\'" + " none" + f" \'{self.afd_im_path}\'")
        os.system('fixel2voxel -number 11' + f" \'{self.peak_amp_path}\'" + " none" + f" \'{self.peak_amp_im_path}\'")

        os.system('fixel2peaks -number 11' + f" \'{self.fixel_directory}\'" + f" \'{self.fixel_directions_path}\'")

        self.create_thresholded_fixels()

    

    def create_1fix_mask(self):
        '''Create the mask of voxels in the corpus callosum which contain only 1 fibre '''
        os.system("mrconvert" + f" \'{self.index_nifti}\'" + " -coord 3 0 " + f" \'{self.index_channel_1_path}\'")
        os.system("mrcalc" + f" \'{self.index_channel_1_path}\'" + " 1 -eq" + f" \'{self.CC_path}\'" + " -mult" + f" \'{self.CC_1fixel_path}\'")
        os.remove(self.index_channel_1_path)
        
    def create_2fix_mask(self):
        '''
        Creates the mask containing only voxels in all of the MCP and CST which contain 2 voxels.
        '''
        os.system("mrconvert" + f" \'{self.index_nifti}\'" + " -coord 3 0 " + f" \'{self.index_channel_1_path}\'")
        os.system("mrcalc" + f" \'{self.CST_left_path}\'" + f" \'{self.CST_right_path}\'" + " -or" + f" \'{self.CST_whole_path}\'")
        os.system(f"mrcalc {self.index_channel_1_path} 2 -eq {self.CST_whole_path} -mult {self.MCP_path} -mult {self.MCP_CST_2fixel_path}")
        os.remove(self.index_channel_1_path)
        
    def create_3fix_mask(self):
        '''
        Creates the mask containing only voxels in all of the CC, CST and SLF which contain 3 voxels.
        '''
        os.system("mrconvert" + f" \'{self.index_nifti}\'" + " -coord 3 0 " + f" \'{self.index_channel_1_path}\'")
        os.system(f"mrcalc {self.SLF_3_left_path} {self.SLF_3_right_path} -or {self.SLF_2_left_path} -or {self.SLF_2_right_path} -or {self.SLF_1_left_path} -or {self.SLF_1_right_path} -or {self.SLF_whole_path} ")
        os.system(f"mrcalc {self.index_channel_1_path} 3 -eq {self.SLF_whole_path} -and {self.CC_path} -and {self.CST_whole_path} -and {self.CC_CST_SLF_3fixel_path}")
        os.remove(self.index_channel_1_path)

    def fill_tract_seg_dir(self):
        os.system("TractSeg -i" + f" \'{self.gt_data_path}\'" + " -o" + f" \'{self.tractseg_dir}\'" + " --bvals" + f" \'{self.gt_bvals_path}\'" + " --bvecs" + f" \'{self.gt_bvecs_path}\'" + " --raw_diffusion_input --csd_type csd_msmt")
        self.create_1fix_mask()
        self.create_2fix_mask()
        self.create_3fix_mask()
    

    def fill_T1w_dir(self):
        os.system("5ttgen fsl" + f" \'{self.T1w_image_path}\'" + f" \'{self.fivettgen_path}\'" + " -nocrop")
        os.system("mrconvert" + f" \'{self.fivettgen_path}\'" + " -coord 3 2" + f" \'{self.wm_mask_path}\'")

    def fill_undersampled_fod_dir(self):
        if os.path.isdir(self.undersampled_fod_dir) == False:
            os.mkdir(self.undersampled_fod_dir)
        d = UndersampleDataset(self.current_subject, self.data_dir)
        d.all_save()

        self.calculate_undersampled_responses()

    def calculate_undersampled_responses(self):
        os.system(f"dwi2response dhollander {self.undersampled_data_path} {self.undersampled_wm_response_path} {self.undersampled_gm_response_path} {self.undersampled_csf_response_path} -fslgrad {self.undersampled_bvecs_path} {self.undersampled_bvals_path}")

    def process_whole_dataset(self):
        for subject in self.subject_list:
            self.set_subject_paths(subject)
            self.fill_T1w_dir()
            self.fill_diffusion_dir()
            self.fill_gt_fixel_dir()
            self.fill_tract_seg_dir()
            self.fill_undersampled_fod_dir()
            self.calc_gt_fod()
    
    def reset_dataset(self):
        self.reset_T1w_dir()
        self.reset_diffusion_dir()

    def reset_T1w_dir(self):
        for subject in self.subject_list:
            self.set_subject_paths(subject)
            self.ifexists_rm(self.fivettgen_path)
            self.ifexists_rm(self.wm_mask_path)

    def reset_diffusion_dir(self):
        for subject in self.subject_list:
            
            self.set_subject_paths(subject)
            diffusion_files = [self.gt_normalised_data_path, 
                                    self.gt_wm_response_path, 
                                    self.gt_gm_response_path, 
                                    self.gt_csf_response_path,
                                    self.gt_wm_fod_path, 
                                    self.gt_gm_fod_path, 
                                    self.gt_csf_fod_path, 
                                    self.gt_whole_fod_path]
            
            for path in diffusion_files:
                self.ifexists_rm(path)
            
            diffusion_folders = [self.fixel_directory, self.tractseg_dir, self.undersampled_fod_dir]

            for path in diffusion_folders:
                self.ifdir_rmtree(path)

    
    def ifexists_rm(self,path):
        if os.path.exists(path):
            os.remove(path)

    def ifdir_rmtree(self,path):
        if os.path.isdir(path):
            shutil.rmtree(path)

class UndersampleDataset():
    def __init__(self, subject, data_dir):
        
        #Initialising the parameters for the dataset class.
        self.subject = subject
        self.data_dir = data_dir

        #Calculating the mask list and keep lists (using this function here will only work when a constant undersampling pattern is used)
        self.keep_list= self.sample_lists()
    
    def data_save(self):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        #Creating the data path and the mask path.
        dwi_path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'normalised_data.nii.gz')
        
        #Loading the data for the subject
        image = nib.load(dwi_path)
        self.head = image.header
        self.aff = image.affine
        self.image = torch.tensor(image.get_fdata())

        im_usamp = nib.Nifti1Image(self.image[:,:,:,self.keep_list].float().detach().numpy(), affine=self.aff)
        save_path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'undersampled_fod', 'data.nii.gz')
        nib.save(im_usamp, save_path)

    def all_save(self):
        self.data_save()
        self.bval_save()
        self.bvec_save()

    def sample_lists(self):
        '''
        A function which returns two lists - the mask_list, which is the q-space volumes to be masked,
        and keep_list for the q-space volumes not to be masked. This will be used in the __getitem__ function 
        of this class.
        '''

        self.bvals = self.bval_extract()
        
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

        mask_ind = [i for i in range(9, len(b1000_list))]
        keep_ind = [i for i in range(9)]
        
        #Uncomment this for the multi-shell version 
        keep_list = torch.tensor([b1000_list[i]for i in keep_ind] +
                                [b2000_list[i]for i in keep_ind] +
                                [b3000_list[i]for i in keep_ind] +
                                b0_list[:3])

        return keep_list
    
    def bval_extract(self):
        '''
        Input:
            subj (str) - The subject whose bvals you want to extract.
            path (str) - The path to where the data is located (subjects location).
        
        Desc:
            A function to extract the bvalues from the file they are located in and to return them as a list of values.
        '''  
        path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'bvals') 
        bvals = open(path, 'r')
        bvals_str = bvals.read()
        bvals = [int(b) for b in bvals_str.split()]
        return bvals

    def bvec_save(self):
        print('Saving bvectors')
        #Undersampled bvector calculation.
        path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'bvecs')
        
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
        
        save_path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'undersampled_fod', 'bvecs')
        
        with open(save_path, 'w') as temp:
            temp.write(bvecs_string)
        print('Finished Saving bvectors')

    def bval_save(self):
        print('Saving bvals')
        ##Bvalues
        new_bvals = [str(self.bvals[ind]) for ind in self.keep_list]
        bvals_string = ' '.join(new_bvals)

        save_path = os.path.join(self.data_dir, self.subject, 'T1w', 'Diffusion', 'undersampled_fod', 'bvals')
        
        with open(save_path,'w') as temp:
            temp.write(bvals_string)
        print('Finished saving bvals')
    


    
dm = data_manager()
dm.set_subject_paths('103818')
dm.fill_undersampled_fod_dir()
# dm.process_whole_dataset()

# d = UndersampleDataset('100206', '/media/duanj/E44441FB4441D0CA/jxb/Projects/data/hcp')
# print(len(d.keep_list))
# d.all_save()