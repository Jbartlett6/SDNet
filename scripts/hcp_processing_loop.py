'''
hcp_processing_loop:
Script to pre-process a list of HCP data
'''
import sys 
import os

sys.path.append(os.path.join(sys.path[0],'..'))
import preprocessing.preprocess_py as preproc

import os

sys.path.append(os.path.join(sys.path[0],'..'))
import preprocessing.preprocess_py as preproc

# proc_subs = ['113821', '523032', '130518', '151021', '130417', '130720', '202113', '188751', '118225',
# '284646', '120111', '123824', '268850', '123117', '368753', '161327', '176845', '159441', '559457', '115017']

proc_subs = ['145127', #, # Subjects to be used for testing. **Inference**
            '147737',
            '174437',
            '178849',
            '318637',
            '581450',
            '130821']

hcp_dir = '/bask/projects/d/duanj-ai-imaging/jxb1336/hcp_96'

usamp_folder_name = 'undersampled_fod'

sampling_pattern = [3,9,9,9]


if __name__ == '__main__':
    for subject in proc_subs:
        
        path = os.path.join(hcp_dir, subject, 'T1w', 'Diffusion')
        preproc.fully_sampled_FOD(path)
        preproc.fixels_and_masks(path)
        preproc.undersampled_FOD(path, usamp_folder_name = usamp_folder_name, sampling_pattern = sampling_pattern)
        preproc.reset_HCP_dir(path)

