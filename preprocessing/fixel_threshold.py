import os 

import nibabel as nib 
import numpy as np

def fixel_threshold(path):
    if os.path.exists(os.path.join(path, 'fixel_directory', 'fixnet_targets')) == False:
        os.mkdir(os.path.join(path, 'fixel_directory', 'fixnet_targets'))
    
    nifti = nib.load(os.path.join(path, 'fixel_directory', 'index.nii.gz'))
    aff = nifti.affine

    fix_im = np.array(nifti.dataobj)
    fix_im[fix_im >= 4] = 4

    nifti_updated = nib.Nifti1Image(fix_im, aff)

    nib.save(nifti_updated, os.path.join(path, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz'))