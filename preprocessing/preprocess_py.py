import dwi_undersample as dwiusamp

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

def undersampled_FOD(path):
    # If the undersampled_fod directory doesn't exist, make it
    if os.path.exists(os.path.join(path, 'undersampled_fod')) == False:
        os.mkdir(os.path.join(path, 'undersampled_fod'))

    UspDset = dwiusamp.UndersampleDataset(path, os.path.join(path, 'undersampled_fod'))
    UspDset.all_save()

    #Normalising the already undersampled data so the maximum value is 1.
    subprocess.run(['dwinormalise', 'individual', os.path.join(path, 'undersampled_fod', 'data.nii.gz'), 
                    os.path.join(path, 'nodif_brain_mask.nii.gz'), os.path.join(path, 'undersampled_fod', 'normalised_data.nii.gz'), 
                    '-fslgrad', os.path.join(path, 'undersampled_fod', 'bvecs'), os.path.join(path, 'undersampled_fod', 'bvals'),
                    '-intensity', '1'])
    
    # Calculating the undersampled FODs
    subprocess.run(['dwi2response', 'dhollander', os.path.join(path, 'undersampled_fod', 'normalised_data.nii.gz'), os.path.join(path, 'undersampled_fod', 'wm_response.txt'),
                    os.path.join(path, 'undersampled_fod', 'gm_response.txt'), os.path.join(path, 'undersampled_fod', 'csf_response.txt'), '-fslgrad',
                    os.path.join(path, 'undersampled_fod', 'bvecs'), os.path.join(path, 'undersampled_fod', 'bvals')])
    
    subprocess.run(['dwi2fod', '-fslgrad', os.path.join(path, 'undersampled_fod', 'bvecs'), os.path.join(path, 'undersampled_fod', 'bvals'),
                 'msmt_csd', os.path.join(path, 'undersampled_fod', 'normalised_data.nii.gz'), os.path.join(path, 'undersampled_fod', 'wm_response.txt'),
                 os.path.join(path, 'undersampled_fod', 'wm.nii.gz'), os.path.join(path, 'undersampled_fod', 'gm_response.txt'), os.path.join(path, 'undersampled_fod', 'gm.nii.gz'),
                 os.path.join(path, 'undersampled_fod', 'csf_response.txt'), os.path.join(path, 'undersampled_fod', 'csf.nii.gz')])
    
    return 0

def fixels_and_masks(path):
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


    
def reset_HCP_dir(path):
    '''
    A utility function for returning a HCP directory to its original state i.e. removing all 
    processing. This script is useful for testing the above processing functions.
    '''
    os.remove(os.path.join(path, 'wm_response.txt'))
    os.remove(os.path.join(path, 'gm_response.txt'))
    os.remove(os.path.join(path, 'csf_response.txt'))

    os.remove(os.path.join(path, 'wmfod.nii.gz'))
    os.remove(os.path.join(path, 'gm.nii.gz'))
    os.remove(os.path.join(path, 'csf.nii.gz'))

    os.remove(os.path.join(path, '..', '5ttgen.nii.gz'))
    os.remove(os.path.join(path, '..', 'white_matter_mask.nii.gz'))
    
    if os.path.exists(path, 'undersampled_fod'):
        shutil.rmtree(os.path.exists(path, 'undersampled_fod'))

    if os.path.exists(path, 'tractseg'):
        shutil.rmtree(os.path.exists(path, 'tractseg'))
    
    if os.path.exists(path, 'fixel_directory'):
        shutil.rmtree(os.path.exists(path, 'fixel_directory'))
    


def mif_to_nifti(mif_path):
    '''
    Converts a mif file at location mif_path (MRtrix3 native file type) to a nifti file
    in the same location and deletes the mif file.
    '''
    assert os.path.exists(mif_path), f"The mif file {mif_path} doesn't exist." 
    nifti_path = ''.join(mif_path.split('.')[:-1])+'.nii.gz'
    subprocess.run(['mrconvert', mif_path, nifti_path])
    os.remove(mif_path)


     	                
if __name__ == '__main__':
    diffusion_dir = '/media/duanj/F/joe/ton_multi/hcp_like/T1w/Diffusion'
    # print(diffusion_dir)
    fully_sampled_FOD(diffusion_dir)
    # fixels_and_masks(diffusion_dir)
    # undersampled_FOD(diffusion_dir)