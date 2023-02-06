import sys
import os 
import argparse
import numpy as np
sys.path.append(os.path.join(sys.path[0],'utils'))
sys.path.append(os.path.join(sys.path[0],'utils','performance_measures'))
import Performance_Metrics as PM
import options
import inference
import data
import nibabel as nib
import inference

if __name__ == '__main__':
    
    ## Input the options for the classes to take as input - options are used for experiment_name
    test_subjects = ['130821',
                    '145127',
                    '147737',
                    '174437',
                    '178849',
                    '318637',
                    '581450']


    experiment_name = 'test_tmp'
    inference_path = os.path.join('.','checkpoints',experiment_name,'inference')
    data_dir = '/media/duanj/F/joe/hcp_2'

    #Include all measures of accuracy in this loop and save them to the appropriate destination
    inf_obj = inference.InferenceClass(data_dir, 'best_model.pth', experiment_name)
    model_performance = PM.ModelPerformance(data_dir,inference_path,test_subjects)

    for subj in test_subjects:
        print('Performing inference for subject: '+subj)
        inf_obj.run_seq(subj)
    
    model_performance.calc_all_performance()
