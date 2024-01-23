'''
hcp_processing_loop:
Script to pre-process a list of HCP data
'''
import preprocess_py as preproc

import os


proc_subs = ['113821', '523032', '130518', '151021', '130417', '130720', '202113', '188751', '118225',
'284646', '120111', '123824', '268850', '123117', '368753', '161327', '176845', '159441', '559457', '115017']

# hcp_dir = 

if __name__ == '__main__':
    for subject in proc_subs:

        path = os.path.join(hcp_dir, subject, 'T1w', 'Diffusion')
        preproc.fully_sampled_FOD(path)
        preproc.fixels_and_masks(path)
        preproc.undersampled_FOD(path)