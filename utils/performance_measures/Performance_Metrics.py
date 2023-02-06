import os
import numpy as np
import nibabel as nib
import pandas as pd
import shutil
    
class ModelPerformance():
    def __init__(self,data_dir, model_inference_dir,subject_list):
        '''
        Initialise general paths and other general attributes for the class
        '''
        self.model_inference_dir = model_inference_dir
        self.data_dir = data_dir
        self.subject_list = subject_list
        self.ROI_names = ['wm','ROI1','ROI2','ROI3', 'ROI4', 'ROI5', 'ROI6']
        

        #Initialising the Directories where the data is going to be stored. 
        self.init_performance_metric_dir()
        self.init_performance_CSV_paths()

    def init_subject_paths(self,subject):
        '''
        For an individual subject update all of the subject specific paths paths.
        '''
        #General Directories
        self.inf_dir = os.path.join(self.model_inference_dir,subject)
        self.gt_diffusion_dir = os.path.join(self.data_dir, subject, 'T1w', 'Diffusion')
        self.tract_seg_masks = os.path.join(self.gt_diffusion_dir, 'tractseg', 'bundle_segmentations')

        #Inf FOD directory
        self.inf_fod = os.path.join(self.inf_dir, 'inf_wm_fod.nii.gz')
    
        #Masks:
        self.wm_mask_path_mif = os.path.join(self.data_dir, subject, 'T1w', 'white_matter_mask.mif')
        self.wm_mask_path = os.path.join(self.data_dir, subject, 'T1w', 'white_matter_mask.nii.gz')
        self.CC_mask_path = os.path.join(self.tract_seg_masks, 'CC.nii.gz')
        self.MCP_mask_path = os.path.join(self.tract_seg_masks, 'MCP.nii.gz')
        self.CST_mask_path = os.path.join(self.tract_seg_masks, 'CST_whole.nii.gz')
        self.onefix_mask_path = os.path.join(self.tract_seg_masks, 'CC_1fixel.nii.gz')
        self.twofix_mask_path = os.path.join(self.tract_seg_masks, 'MCP_CST_2fixel.nii.gz')
        self.threefix_mask_path = os.path.join(self.tract_seg_masks, 'CC_CST_SLF_3fixel.nii.gz')

        self.masks_list = [self.wm_mask_path, self.CC_mask_path, self.MCP_mask_path, self.CST_mask_path, self.onefix_mask_path, self.twofix_mask_path, self.threefix_mask_path]
        #Initialising the fixel paths: (must have updated the inf_dir first)
        self.init_fixel_paths()

        #Initialising sse and acc paths:
        self.sse_path = os.path.join(self.inf_dir, 'sse.nii.gz')
        self.acc_path = os.path.join(self.inf_dir, 'acc.nii.gz')

        #Initialising GT paths:
        self.gt_fixel_directory = os.path.join(self.gt_diffusion_dir, 'fixel_directory')
        self.gt_afd_im_path_mif = os.path.join(self.gt_fixel_directory, 'afd_im.mif')
        self.gt_afd_im_path = os.path.join(self.gt_fixel_directory, 'afd_im.nii.gz')
        self.gt_pa_im_path_mif = os.path.join(self.gt_fixel_directory, 'peak_amp_im.mif')
        self.gt_pa_im_path = os.path.join(self.gt_fixel_directory, 'peak_amp_im.nii.gz')
        self.gt_fod_path = os.path.join(self.gt_diffusion_dir, 'wmfod.nii.gz')
        self.gt_index_path = os.path.join(self.gt_fixel_directory, 'index.nii.gz')

    ### Path Initialisation Methods ###
    def init_performance_metric_dir(self):
        '''
        Creating the directories where the csv's for individual performance metrics will be stored.
        '''
        self.performance_metric_dir = os.path.join(self.model_inference_dir, 'Performance_Metrics')
        
        #Creating the overall performance metric folder.
        if os.path.isdir(self.performance_metric_dir) == False:
                os.mkdir(self.performance_metric_dir)

    def init_performance_CSV_paths(self):
        '''
        Initialising the CSV paths which will be used to store each of the individual performance metrics.
        '''
        self.AFDE_csv_path = os.path.join(self.performance_metric_dir, 'AFDE.csv')
        self.PAE_csv_path = os.path.join(self.performance_metric_dir, 'PAE.csv')
        self.MAE_csv_path = os.path.join(self.performance_metric_dir, 'MAE.csv')
        self.fixel_accuracy_csv_path = os.path.join(self.performance_metric_dir, 'Fixel_Accuracy.csv')

        self.SSE_csv_path = os.path.join(self.performance_metric_dir, 'SSE.csv')
        self.ACC_csv_path = os.path.join(self.performance_metric_dir, 'ACC.csv')

    def init_fixel_paths(self):
        '''
        Given the inference directory define all paths which relate to the fixel directory.

        Subject specific paths should be initialised before calling this function.
        '''
        self.fixel_directory = os.path.join(self.inf_dir, 'fixel_directory')
        self.index_mif = os.path.join(self.fixel_directory, 'index.mif')
        self.index_nifti = os.path.join(self.fixel_directory, 'index.nii.gz')
        self.fix_err_path = os.path.join(self.fixel_directory, 'fix_err.nii.gz')

        self.afd_path = os.path.join(self.fixel_directory, 'afd.nii.gz')
        self.afd_im_path = os.path.join(self.fixel_directory, 'afd_im.nii.gz')

        self.peak_amp_path = os.path.join(self.fixel_directory, 'peak_amp.nii.gz')
        self.peak_amp_im_path = os.path.join(self.fixel_directory, 'peak_amp_im.nii.gz')

        self.afde_path = os.path.join(self.fixel_directory, 'afde.nii.gz')
        self.pae_path = os.path.join(self.fixel_directory, 'pae.nii.gz')

    def create_subject_fixels(self):
        '''
        If the fixel directory doesn't exist then this function will create the fixel directory.

        Subject specific paths should be initialised before calling this function.
        '''

        #Creating the fixel directory and converting the index function to a nifti image.
        if os.path.exists(self.fixel_directory):
            shutil.rmtree(self.fixel_directory)

        os.system('fod2fixel -force -afd afd.nii.gz -peak_amp peak_amp.nii.gz ' + f"\'{self.inf_fod}\'" + ' ' + f"\'{self.fixel_directory}\'")

        #Converting the index file from .mif to .nifti
        if os.path.exists(self.index_nifti) == False:
            os.system('mrconvert ' + f"\'{self.index_mif}\'" + ' ' + f"\'{self.index_nifti}\'")

        #Removing the uneccessary index.mif file
        if os.path.exists(self.index_mif):
            os.remove(self.index_mif)
        

        #Creating the afd and peak amplitude images.
        #Calculating the scalar fixel based analysis comparisons
        os.system('fixel2voxel -force -number 11 ' + f"\'{self.afd_path}\'" + ' none ' + f"\'{self.afd_im_path}\'")
        os.system('fixel2voxel -force -number 11 ' + f"\'{self.peak_amp_path}\'" + ' none ' + f"\'{self.peak_amp_im_path}\'")

        

    def allsub_fixels(self):
        for i, subject in enumerate(self.subject_list):
            self.init_subject_paths(subject)
            self.create_subject_fixels()
    
    def check_wm_mask(self):
        '''
        If the white matter mask only exists in .mif format then the the mif image is converted to nifti
        image. 

        Subject specific paths should be initialised before calling this function.
        '''
        if os.path.exists(self.wm_mask_path) == False:
            os.system(f'mrconvert {self.wm_mask_path_mif} {self.wm_mask_path}')
    
    ### AFDE functions###
    def calc_afde_image(self):
        '''
        Calculates the AFDE image for the current subject over a range of regions of interest.
        The AFDE image is then saved in the inference path as AFDE.nii.gz
        '''
        inf_afd_nib = nib.load(self.afd_im_path)
        gt_afd_nib = nib.load(self.gt_afd_im_path)

        inf_afd_np = np.array(inf_afd_nib.dataobj)
        gt_afd_np = np.array(gt_afd_nib.dataobj)

        afde_np = np.sum(np.abs(inf_afd_np - gt_afd_np), axis = 3)
        afde_nifti = nib.Nifti1Image(afde_np, affine = inf_afd_nib.affine)
        nib.save(afde_nifti, self.afde_path)

    def afde_ROI_averages(self):
        '''
        This method calculates the average AFDE in the ROIs given the current subject attributes.
        '''
        afde_nifti = nib.load(self.afde_path)
        afde_np = np.array(afde_nifti.dataobj)
        afde_ROIs = []
        for mask_path in self.masks_list:
            mask_np = self.load_mask(mask_path)
            afde_ROIs.append(np.mean(afde_np[mask_np[:,:,:] > 0.5]))

        return afde_ROIs

    def calc_allsub_AFDE(self):
        '''
        For all subjects in self.subject list this method calculates the average AFDE over all regions of interest, 
        saves them in the CSV file.
        '''
        AFDE_array = np.zeros((len(self.subject_list), len(self.ROI_names)))
        for i, subject in enumerate(self.subject_list):
            
            self.init_subject_paths(subject)
            #If the afde image doesn't exist for the subject calculate it:
            if os.path.exists(self.afde_path) == False:
                self.calc_afde_image()
            #Calculate the AFDE for the ROI averages and add them to the array.    
            AFDE_array[i,:] = self.afde_ROI_averages()

        AFDE_df = pd.DataFrame(AFDE_array, columns=self.ROI_names)
        AFDE_df.to_csv(self.AFDE_csv_path, sep = ',')
    ###PAE functions###
    def calc_pae_image(self):
        '''
        Calculates the AFDE image for the current subject over a range of regions of interest.
        The AFDE image is then saved in the inference path as AFDE.nii.gz
        '''
        inf_peak_amp_nib = nib.load(self.peak_amp_im_path)
        gt_peak_amp_nib = nib.load(self.gt_pa_im_path)

        inf_peak_amp_np = np.array(inf_peak_amp_nib.dataobj)
        gt_peak_amp_np = np.array(gt_peak_amp_nib.dataobj)

        pae_np = np.sum(np.abs(inf_peak_amp_np - gt_peak_amp_np), axis = 3)
        pae_nifti = nib.Nifti1Image(pae_np, affine = inf_peak_amp_nib.affine)
        nib.save(pae_nifti, self.pae_path)
    
    def pae_ROI_averages(self):
        '''
        This method calculates the average PAE in the ROIs given the current subject attributes.
        '''
        pae_nifti = nib.load(self.pae_path)
        pae_np = np.array(pae_nifti.dataobj)
        pae_ROIs = []
        for mask_path in self.masks_list:
            mask_np = self.load_mask(mask_path)
            pae_ROIs.append(np.mean(pae_np[mask_np == 1]))

        return pae_ROIs
    
    def calc_allsub_PAE(self):
        '''
        For all subjects in self.subject list this method calculates the average AFDE over all regions of interest, 
        saves them in the CSV file.
        '''
        PAE_array = np.zeros((len(self.subject_list), len(self.ROI_names)))
        for i, subject in enumerate(self.subject_list):
            
            self.init_subject_paths(subject)
            #If the afde image doesn't exist for the subject calculate it:
            if os.path.exists(self.pae_path) == False:
                self.calc_pae_image()
            #Calculate the AFDE for the ROI averages and add them to the array.    
            PAE_array[i,:] = self.pae_ROI_averages()

        PAE_df = pd.DataFrame(PAE_array, columns=self.ROI_names)
        PAE_df.to_csv(self.PAE_csv_path, sep = ',')
    ###SSE functions###
    def calc_sse_image(self):
        '''
        Calculates the SSE image for the current subject over a range of regions of interest.
        The AFDE image is then saved in the inference path as AFDE.nii.gz
        '''
        inf_fod_nib = nib.load(self.inf_fod)
        gt_fod_nib = nib.load(self.gt_fod_path)


        inf_fod_np = np.array(inf_fod_nib.dataobj)
        gt_fod_np = np.array(gt_fod_nib.dataobj)

        sse_np = np.sum((inf_fod_np - gt_fod_np)**2, axis = 3)
        sse_nifti = nib.Nifti1Image(sse_np, affine = inf_fod_nib.affine)
        nib.save(sse_nifti, self.sse_path)

    def sse_ROI_averages(self):
        '''
        This method calculates the average SSE in the ROIs given the current subject attributes.
        '''
        sse_nifti = nib.load(self.sse_path)
        sse_np = np.array(sse_nifti.dataobj)
        sse_ROIs = []
        for mask_path in self.masks_list:
            mask_np = self.load_mask(mask_path)
            sse_ROIs.append(np.mean(sse_np[mask_np == 1]))

        return sse_ROIs
    
    def calc_allsub_SSE(self):
        '''
        For all subjects in self.subject list this method calculates the average AFDE over all regions of interest, 
        saves them in the CSV file.
        '''
        SSE_array = np.zeros((len(self.subject_list), len(self.ROI_names)))
        for i, subject in enumerate(self.subject_list):
            
            self.init_subject_paths(subject)
            #If the afde image doesn't exist for the subject calculate it:
            if os.path.exists(self.sse_path) == False:
                self.calc_sse_image()
            #Calculate the AFDE for the ROI averages and add them to the array.    
            SSE_array[i,:] = self.sse_ROI_averages()

        SSE_df = pd.DataFrame(SSE_array, columns=self.ROI_names)
        SSE_df.to_csv(self.SSE_csv_path, sep = ',')
    
    ###ACC functions###
    def calc_acc_image(self):
        '''
        Calculates the SSE image for the current subject over a range of regions of interest.
        The AFDE image is then saved in the inference path as AFDE.nii.gz
        '''
        #Loading the nifti images in.
        inf_fod_nib = nib.load(self.inf_fod)
        gt_fod_nib = nib.load(self.gt_fod_path)

        #Converting the nifti images to numpy images.
        inf_fod_np = np.array(inf_fod_nib.dataobj)
        gt_fod_np = np.array(gt_fod_nib.dataobj)

        #Calculating the ACC image in numpy.
        acc_dot = inf_fod_np[:, :, :, 1:45] * gt_fod_np[:, :, :, 1:45]
        print(acc_dot.shape)
        acc_numerator = np.sum(acc_dot, axis = 3)
        acc_denominator = np.linalg.norm(inf_fod_np[:, :, :, 1:45],axis = 3) * np.linalg.norm(gt_fod_np[:, :, :, 1:45], axis=3)
        
        acc_np = np.divide(acc_numerator,acc_denominator)

        #Saving the ACC image as a nifti file.
        acc_nifti = nib.Nifti1Image(acc_np, affine = inf_fod_nib.affine)
        nib.save(acc_nifti, self.acc_path)

    def acc_ROI_averages(self):
        '''
        This method calculates the average ACC in the ROIs given the current subject attributes.
        '''
        acc_nifti = nib.load(self.acc_path)
        acc_np = np.array(acc_nifti.dataobj)

        acc_ROIs = []
        for mask_path in self.masks_list:
            mask_np = self.load_mask(mask_path)
            acc_ROIs.append(np.nanmean(acc_np[mask_np[:,:,:] >= 0.5]))
        
        return acc_ROIs

    def calc_allsub_acc(self):
        '''
        For all subjects in self.subject list this method calculates the average ACC over all regions of interest, 
        saves them in the CSV file.
        '''
        ACC_array = np.zeros((len(self.subject_list), len(self.ROI_names)))
        for i, subject in enumerate(self.subject_list):
            self.init_subject_paths(subject)
            #Only need to check wm mask once in the first case of it being used
            self.check_wm_mask()
            self.check_gt_fix_img()
            #If the afde image doesn't exist for the subject calculate it:
            if os.path.exists(self.acc_path) == False:
                self.calc_acc_image()
            #Calculate the AFDE for the ROI averages and add them to the array.    
            ACC_array[i,:] = self.acc_ROI_averages()

        ACC_df = pd.DataFrame(ACC_array, columns=self.ROI_names)
        ACC_df.to_csv(self.ACC_csv_path, sep = ',')

    ###Fixel Accuracy functions ###
    def calc_fix_err_image(self):
        '''
        Calculates the SSE image for the current subject over a range of regions of interest.
        The AFDE image is then saved in the inference path as AFDE.nii.gz
        '''
        #Loading the nifti images in.
        inf_index_nib = nib.load(self.index_nifti)
        gt_index_nib = nib.load(self.gt_index_path)

        #Converting the nifti images to numpy images.
        inf_index_np = np.array(inf_index_nib.dataobj)
        gt_index_np = np.array(gt_index_nib.dataobj)

        #Calculating the fixel error image in numpy.
        fix_err_np = np.absolute(inf_index_np[:,:,:,0] - gt_index_np[:,:,:,0])
        

        #Saving the ACC image as a nifti file.
        fix_err_nifti = nib.Nifti1Image(fix_err_np, affine = inf_index_nib.affine)
        nib.save(fix_err_nifti, self.fix_err_path)

    def fix_err_ROI_averages(self):
        '''
        This method calculates the average fixel error in the ROIs given the current subject attributes.
        '''
        fix_err_nifti = nib.load(self.fix_err_path)
        fix_err_np = np.array(fix_err_nifti.dataobj)
        fix_err_ROIs = []
        for mask_path in self.masks_list:
            mask_np = self.load_mask(mask_path)
            
            fix_accuracy_numerator = np.sum(fix_err_np[mask_np == 1] == 0)
            fix_accuracy_denominator = np.sum(mask_np == 1)
            
            fix_err_ROIs.append(fix_accuracy_numerator/fix_accuracy_denominator)

        return fix_err_ROIs

    def calc_allsub_fix_err(self):
        '''
        For all subjects in self.subject list this method calculates the average fixel error over all regions of interest, 
        saves them in the CSV file.
        '''
        fix_err_array = np.zeros((len(self.subject_list), len(self.ROI_names)))
        for i, subject in enumerate(self.subject_list):
            
            self.init_subject_paths(subject)
            #If the afde image doesn't exist for the subject calculate it:
            if os.path.exists(self.fix_err_path) == False:
                self.calc_fix_err_image()
            #Calculate the AFDE for the ROI averages and add them to the array.    
            fix_err_array[i,:] = self.fix_err_ROI_averages()

            fix_err_df = pd.DataFrame(fix_err_array, columns=self.ROI_names)
            fix_err_df.to_csv(self.fixel_accuracy_csv_path, sep = ',')
    
    def load_mask(self, mask_path):
        '''
        Loads the mask_path into a numpy array

        If the mask has too many dimensions then the fourth dimension is removed

        If the mask is a white matter mask the it must be flipped in the first axis (since FSL
        changes the stride from -1 to 1)
        '''
        mask_nifti = nib.load(mask_path)
        mask_np = np.array(mask_nifti.dataobj)
            #Masks calculated by tractseg have an extra dimensions which needs removing
        if mask_np.ndim == 4:
            mask_np = mask_np[:,:,:,0]
        
        #Flipping the image in the first dimension if its the white matter mask (required to get the same results as MRTrix3)
        if mask_path == self.wm_mask_path:
            mask_np = mask_np[::-1,:,:]
        
        return mask_np

    def check_gt_fix_img(self):
        '''
        Converts the ground truth fixel paths from .mif files to .nifti files. The create 
        subject fixels may do this already. 
        '''
        if os.path.exists(self.gt_afd_im_path) == False:
            os.system(f'mrconvert {self.gt_afd_im_path_mif} {self.gt_afd_im_path}')

        if os.path.exists(self.gt_pa_im_path) == False:
            os.system(f'mrconvert {self.gt_pa_im_path_mif} {self.gt_pa_im_path}')

    def allsub_preprocessing(self):
        '''
        Performs the neccessaary preprocessing steps for all subjects, making sure all necessary images exist
        prior to calculating performance metrics.
        '''
        #Make sure all of the appropriate images have been calculated 
        for i, subject in enumerate(self.subject_list):
            self.init_subject_paths(subject)
            
            self.create_subject_fixels()
            self.check_wm_mask()
            self.check_gt_fix_img()

      
    def calc_all_performance(self):
        '''
        This function calculates, and updates all of the perfromance metrics to be stored in the CSV files in the performance metrics folder.
        The number of subjects and which ROIs are dictated by the self.ROI_names and self.subject_list attributes defined in self.__init__().
        '''
        self.allsub_fixels()
        self.allsub_preprocessing()
        self.calc_allsub_acc()
        self.calc_allsub_AFDE()
        self.calc_allsub_PAE()
        self.calc_allsub_SSE()
        self.calc_allsub_fix_err()
