'''
hcp_processing_loop:
Script to pre-process a list of HCP data (proc_subs)
'''
import sys 
sys.path.append('/home/jxb1336/code/project_1/SDNet/SDNet')
import preprocessing.preprocess_py as preproc

import os

proc_subs = ['145127']

hcp_dir = '/media/duanj/F/joe/hcp_2'

def list_incomplete_downloads(proc_subs):
    checklist = []
    for subject in proc_subs:
        path = os.path.join('/bask/projects/d/duanj-ai-imaging/jxb1336/hcp', subject, 'T1w', 'Diffusion')
        check_bool = preproc.folders_check(path)
        if check_bool == False:
            checklist.append(subject)
    return checklist

if __name__ == '__main__':

    for subject in proc_subs:
        print(subject)
        path = os.path.join(hcp_dir, subject, 'T1w', 'Diffusion')
        
        # preproc.reset_HCP_dir(path)
        # preproc.fully_sampled_FOD(path)
        # preproc.fixels_and_masks(path)
        # preproc.undersampled_FOD(path)
        preproc.preprocessing_test(path)
        # preproc.HCP_download_test(path)
