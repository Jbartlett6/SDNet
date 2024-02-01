'''
A collection of functions to be used for preprocessing, and manipulating resetting the HCP directories and testing 
that preprocessing has been run correctly. 

    fully_sampled_FOD - Calculate the fully sampled FODs
    
    undersampled_FOD - Undersampled the data (data.nii.gz, bvecs and bvals) and calculate the subsequent FODs. 
    
    fixels_and_mask - Populating the fixel directry for the fully sampled FODs and the masks which will be used to evaluate performance
    as well as ensure only white matter and grey matter voxels are used in training. Tractseg is used in this 
    function. 

    reset_HCP_dir - Removes the files and directories that are created in teh HCP subject's folder due to preprocessing. 

    HCP_download_test - Tests that all of the files that are required to perform preprocessing can be found in the directory.

    preprocessing_test - Check that preprocessing has een performed correctly and that the data necessary to run 
    training and test are available. 
'''
import sys 
import os
sys.path.append(os.path.join(sys.path[0],'..'))
import preprocessing.dwi_undersample as dwiusamp
import preprocessing.fixel_threshold as fixel_threshold

import subprocess 
import os
import shutil

def fully_sampled_FOD(path):
    # Fully sampled FOD
    subprocess.run(['dwi2response', 'dhollander', os.path.join(path, 'data.nii.gz'), 
                    os.path.join(path, 'wm_response.txt'), os.path.join(path, 'gm_response.txt'), 
                    os.path.join(path, 'csf_response.txt'), '-fslgrad', 
                    os.path.join(path, 'bvecs'), os.path.join(path, 'bvals')])
    
    
    subprocess.run(['dwi2fod', '-fslgrad', os.path.join(path, 'bvecs'), os.path.join(path, 'bvals'),
                    'msmt_csd', os.path.join(path, 'data.nii.gz'), os.path.join(path, 'wm_response.txt'),
                    os.path.join(path, 'wmfod.nii.gz'), os.path.join(path, 'gm_response.txt'), os.path.join(path, 'gm.nii.gz'), 
                    os.path.join(path, 'csf_response.txt'), os.path.join(path, 'csf.nii.gz')])

    subprocess.run(['mrcat', '-axis', '3', os.path.join(path, 'wmfod.nii.gz'), os.path.join(path, 'gm.nii.gz'), os.path.join(path, 'csf.nii.gz'), os.path.join(path, 'gt_fod.nii.gz')])

    return 0

def undersampled_FOD(path, usamp_folder_name = 'undersampled_fod', sampling_pattern = [3,9,9,9]):
    # If the undersampled_fod directory doesn't exist, make it
    if os.path.exists(os.path.join(path, usamp_folder_name)) == False:
        os.mkdir(os.path.join(path, usamp_folder_name))

    UspDset = dwiusamp.UndersampleDataset(path, os.path.join(path, usamp_folder_name), sampling_pattern = sampling_pattern)
    UspDset.all_save()

    #Normalising the already undersampled data so the maximum value is 1.
    subprocess.run(['dwinormalise', 'individual', os.path.join(path, usamp_folder_name, 'data.nii.gz'), 
                    os.path.join(path, 'nodif_brain_mask.nii.gz'), os.path.join(path, usamp_folder_name, 'normalised_data.nii.gz'), 
                    '-fslgrad', os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals'),
                    '-intensity', '1'])
    
    # Calculating the undersampled FODs
    subprocess.run(['dwi2response', 'dhollander', os.path.join(path, usamp_folder_name, 'normalised_data.nii.gz'), os.path.join(path, usamp_folder_name, 'wm_response.txt'),
                    os.path.join(path, usamp_folder_name, 'gm_response.txt'), os.path.join(path, usamp_folder_name, 'csf_response.txt'), '-fslgrad',
                    os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals')])
    
    subprocess.run(['dwi2fod', '-fslgrad', os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals'),
                 'msmt_csd', os.path.join(path, usamp_folder_name, 'normalised_data.nii.gz'), os.path.join(path, usamp_folder_name, 'wm_response.txt'),
                 os.path.join(path, usamp_folder_name, 'wm.nii.gz'), os.path.join(path, usamp_folder_name, 'gm_response.txt'), os.path.join(path, usamp_folder_name, 'gm.nii.gz'),
                 os.path.join(path, usamp_folder_name, 'csf_response.txt'), os.path.join(path, usamp_folder_name, 'csf.nii.gz')])
    
    return 0

def fixels_and_masks(path):
    
    def mif_to_nifti(mif_path):
        '''
        Converts a mif file at location mif_path (MRtrix3 native file type) to a nifti file
        in the same location and deletes the mif file.
        '''
        assert os.path.exists(mif_path), f"The mif file {mif_path} doesn't exist." 
        nifti_path = ''.join(mif_path.split('.')[:-1])+'.nii.gz'
        subprocess.run(['mrconvert', mif_path, nifti_path])
        os.remove(mif_path)

        return 0
    
    # 5 tissue segmentation
    subprocess.run(['5ttgen', 'fsl', os.path.join(path, '..', 'T1w_acpc_dc_restore_1.25.nii.gz'), os.path.join(path, '..', '5ttgen.nii.gz'), '-nocrop'])
    subprocess.run(['mrconvert', os.path.join(path, '..', '5ttgen.nii.gz'), '-coord', '3', '2', os.path.join(path,'..','white_matter_mask.nii.gz')])

    # FOD segmentation
    subprocess.run(['fod2fixel', '-afd', 'afd.mif', '-peak_amp', 'peak_amp.mif', os.path.join(path, 'wmfod.nii.gz'), 
                    os.path.join(path, 'fixel_directory')])
    
    # Converting the mif fixel files to nifti files
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'afd.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'peak_amp.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'index.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'directions.mif'))

    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'peak_amp.nii.gz'), 
                    'none', os.path.join(path, 'fixel_directory', 'peak_amp_im.nii.gz')])
    
    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'afd.nii.gz'), 'none', 
                    os.path.join(path, 'fixel_directory', 'afd_im.nii.gz')])

    fixel_threshold.fixel_threshold(path)    
    
    # Tractseg
    subprocess.run(['TractSeg', '-i', os.path.join(path, 'data.nii.gz'), '-o', os.path.join(path, 'tractseg'), 
                    '--bvals', os.path.join(path, 'bvals'), '--bvecs', os.path.join(path, 'bvecs'), '--raw_diffusion_input',
                    '--csd_type', 'csd_msmt'])

    # Extracting the number of fixels from the index image.
    subprocess.run(['mrconvert', os.path.join(path, 'fixel_directory', 'index.nii.gz'), '-coord', '3', '0', os.path.join(path, 'fixel_directory', 'index_1.nii.gz')])
    
    # CC containing 1 fixel 
    subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '1', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC.nii.gz'), 
                    '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_1fixel.nii.gz')])

    # MCP CST intersection containing 2 fixels 
    subprocess.run(['mrcalc', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_left.nii.gz'), os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_right.nii.gz'),
                    '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz')])
    subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '2', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz'),
                    '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP.nii.gz'), '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP_CST_2fixel.nii.gz')])

    # CC, CST and SLF intersection containing 3 fixels
    subprocess.run(['mrcalc', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_III_left.nii.gz'), os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_III_right.nii.gz'),
                    '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_II_left.nii.gz'), '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_II_right.nii.gz'),
                    '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_I_left.nii.gz'), '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_I_right.nii.gz'),
                    '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_whole.nii.gz')])
    
    subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '3', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_whole.nii.gz'),
                    '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC.nii.gz'), '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz'),
                    '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_CST_SLF_3fixel.nii.gz')])
    
    os.remove(os.path.join(path, 'fixel_directory', 'index_1.nii.gz'))

    return 0
    
def reset_HCP_dir(path, usamp_folder_name = 'undersampled_fod'):
    '''
    A utility function for returning a HCP directory to its original state i.e. removing all 
    processing. This script is useful for testing the above processing functions.
    '''
    print(f'Resetting {path}')

    def custom_rm(path_rm):
        if os.path.exists(path_rm):
            os.remove(path_rm)
        
    custom_rm(os.path.join(path, 'wm_response.txt'))
    custom_rm(os.path.join(path, 'gm_response.txt'))
    custom_rm(os.path.join(path, 'csf_response.txt'))

    custom_rm(os.path.join(path, 'wmfod.nii.gz'))
    custom_rm(os.path.join(path, 'gm.nii.gz'))
    custom_rm(os.path.join(path, 'csf.nii.gz'))

    custom_rm(os.path.join(path, '..', '5ttgen.nii.gz'))
    custom_rm(os.path.join(path, '..', 'white_matter_mask.nii.gz'))
    
    if os.path.exists(os.path.join(path, usamp_folder_name)):
        shutil.rmtree(os.path.join(path, usamp_folder_name))

    if os.path.exists(os.path.join(path, 'tractseg')):
        shutil.rmtree(os.path.join(path, 'tractseg'))
    
    if os.path.exists(os.path.join(path, 'fixel_directory')):
        shutil.rmtree(os.path.join(path, 'fixel_directory'))

    print(f'Finished resetting {path}')

    return 0
    
def HCP_download_test(path):
    folders_present = (os.path.exists(os.path.join(path, '..', 'T1w_acpc_dc_restore_1.25.nii.gz'))
    and os.path.exists(os.path.join(path, 'bvecs'))
    and os.path.exists(os.path.join(path, 'bvals'))
    and os.path.exists(os.path.join(path, 'data.nii.gz'))
    and os.path.exists(os.path.join(path, 'nodif_brain_mask.nii.gz')))

    print(folders_present)

    return folders_present

def preprocessing_test(path, usamp_folder_name = 'undersampled_fod'):
    '''
    Function to test that pre-processing has been performed and all of the correct files have been created. 
    For the given path each folder is checked that it contains the files that should have been calculated 
    during the pre-processing. The output is a tuple, the first is whether the files exist to use the 
    subject for training, and the second whether all of the necessary files exist. 
    '''
    assert os.path.exists(path), "The path being tested for does not exist"

    T1w_bool = (os.path.exists(os.path.join(path, '..', '5ttgen.nii.gz'))
                   )

    diffusion_bool = (os.path.exists(os.path.join(path, 'wm_response.txt'))
                         and os.path.exists(os.path.join(path, 'gm_response.txt'))
                         and os.path.exists(os.path.join(path, 'csf_response.txt'))
                         and os.path.exists(os.path.join(path, 'wmfod.nii.gz'))
                         and os.path.exists(os.path.join(path, 'gm.nii.gz'))
                         and os.path.exists(os.path.join(path, 'csf.nii.gz'))
                        and os.path.exists(os.path.join(path, 'gt_fod.nii.gz'))
                        )
    
    fixel_directory_train_bool = (os.path.exists(os.path.join(path, 'fixel_directory', 'index.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'fixnet_targets'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz'))
                       )

    fixel_directory_test_bool = (os.path.exists(os.path.join(path, 'fixel_directory', 'index.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'afd_im.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'peak_amp_im.nii.gz'))
                       )
    
    undersampled_fod_train_bool = (os.path.exists(os.path.join(path, usamp_folder_name, 'bvals'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'bvecs'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'data.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'normalised_data.nii.gz'))
                        )
    
    undersampled_fod_test_bool = (os.path.exists(os.path.join(path, usamp_folder_name, 'wm_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'gm_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'csf_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'wm.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'gm.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'csf.nii.gz'))
                        )

    tractseg_bool = (os.path.exists(os.path.join(path, 'tractseg'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'peaks.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_1fixel.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP_CST_2fixel.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_CST_SLF_3fixel.nii.gz'))
                    )

    training_bool = diffusion_bool and undersampled_fod_train_bool and T1w_bool and fixel_directory_train_bool
    training_and_testing_bool = training_bool and tractseg_bool and fixel_directory_test_bool

    report =f'''
            Report for path: {path} \n
            *** TRAINING STATUS ***\n
            T1w folder status: {T1w_bool} \n
            Diffusion status: {diffusion_bool} \n
            Fixel directory status: {fixel_directory_train_bool}\n
            Undersampled FOD status: {undersampled_fod_train_bool}\n\n

            *** TESTING STATUS ***\n
            Fixel directory status: {fixel_directory_test_bool}\n
            Undersampled FOD status: {undersampled_fod_test_bool}\n
            Tractseg status: {tractseg_bool}\n\n

            TRAINING STATUS: {training_bool}
            TRAINING AND TESTING STATUS: {training_and_testing_bool}\n
            '''
    print(report) 

    return training_bool, training_and_testing_bool

     	                
if __name__ == '__main__':

    diffusion_dir = '/mnt/d/Diffusion_data/Tong/Tong_as_HCP'
    usamp_folder_name = 'undersampled_92'
    preprocessing_test(diffusion_dir)
    print(diffusion_dir)
    fully_sampled_FOD(diffusion_dir)
    fixels_and_masks(diffusion_dir)
    undersampled_FOD(diffusion_dir, usamp_folder_name = usamp_folder_name, sampling_pattern=[6,30,30,30])