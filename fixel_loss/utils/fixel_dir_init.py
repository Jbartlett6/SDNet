#This script creates the fixel directory as used throughout the SDNet scripts for the list of subejcts specified.
import os

subject_list =  [100206,
100307,
100408,
100610,
101006,
101107,
101309,
101915,
102008,
102109,
102311,
102513,
102614,
102715,
102816,
103010,
103111,
103212,
103414,
103515,
103818]

data_dir = '/media/duanj/F/joe/hcp_2'


for subject in subject_list:
    subj_dir = os.path.join(data_dir, str(subject), 'T1w', 'Diffusion')
    gt_fod = os.path.join(subj_dir, 'wmfod.nii.gz')
    gt_fixel_dir = os.path.join(subj_dir, 'fixel_directory')
    
    index_mif = os.path.join(gt_fixel_dir, 'index.mif')
    index_nifti = os.path.join(gt_fixel_dir, 'index.nii.gz')
    
    # os.system('fod2fixel -afd afd.mif -peak_amp peak_amp.mif ' + gt_fod + ' ' + gt_fixel_dir)
    # os.system('mrconvert ' + index_mif + ' ' + index_nifti)
    os.system('mrstats ' + index_mif)
