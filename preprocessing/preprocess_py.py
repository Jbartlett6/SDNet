import dwi_undersample as dwiusamp

import subprocess 
import os

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
    
    # 5 tissue segmentation
    subprocess.run(['5ttgen', 'fsl', os.path.join(path, '..', 'T1w_acpc_dc_restore_1.25.nii.gz'), os.path.join(path, '..', '5ttgen.nii.gz'), '-nocrop'])
    subprocess.run(['mrconvert', os.path.join(path, '..', '5ttgen.nii.gz'), '-coord', '3', '2', os.path.join(path,'..','white_matter_mask.nii.gz')])

    # FOD segmentation
    subprocess.run(['fod2fixel', '-afd', 'afd.mif', '-peak_amp', 'peak_amp.mif', os.path.join(path, 'wmfod.nii.gz'), 
                    os.path.join(path, 'fixel_directory')])
    
    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'peak_amp.mif'), 
                    'none', os.path.join(path, 'fixel_directory', 'peak_amp_im.mif')])
    
    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'afd.mif'), 'none', 
                    os.path.join(path, 'fixel_directory', 'afd_im.mif')])

    # Tractseg
    subprocess.run(['TractSeg', '-i', os.path.join(path, 'data.nii.gz'), '-o', os.path.join(path, 'tractseg'), 
                    '--bvals', os.path.join(path, 'bvals'), '--bvecs', os.path.join(path, 'bvecs'), '--raw_diffusion_input',
                    '--csd_type', 'csd_msmt'])

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


    
def reset_HCP_dir(path):

    os.remove(os.path.join(path, 'wm_response.txt'))
    os.remove(os.path.join(path, 'gm_response.txt'))
    os.remove(os.path.join(path, 'csf_response.txt'))

    os.remove(os.path.join(path, 'wmfod.nii.gz'))
    os.remove(os.path.join(path, 'gm.nii.gz'))
    os.remove( os.path.join(path, 'csf.nii.gz'))
     	                
if __name__ == '__main__':
    diffusion_dir = '/media/duanj/F/joe/hcp_2/100206_copy/T1w/Diffusion'
    print(diffusion_dir)
    # fully_sampled_FOD(diffusion_dir)
    undersampled_FOD(diffusion_dir)