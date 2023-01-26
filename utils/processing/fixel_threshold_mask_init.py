import os
import numpy as np
import nibabel as nib
import shutil

#Creating the thresholded fixel images for the lassification network. 
# train_subject_list = ['100206',
# '100307',
# '100408',
# '100610',
# '101006',
# '101107',
# '101309',
# '101915',
# '102109',
# '102311',
# '102513',
# '102614',
# '102715',
# '102816',
# '103010',
# '103111',
# '103212',
# '103414',
# '103515',
# '103818',
# '104012',
# '104416',
# '104820']

# train_subject_list = ['174437',
# '318637',
# '581450',
# '145127',
# '147737',
# '178849',
# '130821']

train_subject_list = ['130821']
data_dir = '/bask/projects/d/duanj-ai-imaging/jxb1336/hcp'


for subject in train_subject_list:
    # fod = os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'wmfod.nii.gz')
    fixel_directory = os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory')
    # if os.path.isdir(fixel_directory):
    #     shutil.rmtree(fixel_directory)
    # os.system('fod2fixel -force -afd afd.mif -peak_amp peak_amp.mif ' + fod + ' ' + fixel_directory)
    # os.system('mrconvert ' + os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'index.mif') + ' ' +os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'index.nii.gz'))
    gt_path = os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'index.nii.gz')
    if not os.path.isdir(os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'fixnet_targets')):
        os.mkdir(os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'fixnet_targets'))
    save_path = os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'fixel_directory', 'fixnet_targets','gt_threshold_fixels.nii.gz')

    