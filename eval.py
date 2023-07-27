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
    # Loading Network Options
    opts = options.network_options()

    inference_path = os.path.join('.','checkpoints',opts.experiment_name,'inference')

    #Include all measures of accuracy in this loop and save them to the appropriate destination
    inf_obj = inference.InferenceClass('best_model.pth', opts.experiment_name, opts)
    # model_performance = PM.ModelPerformance(opts.data_dir, inference_path, opts.test_subject_list)

    for subj in opts.test_subject_list:
        print('Performing inference for subject: '+subj)
        inf_obj.run_seq(subj)
    
    # model_performance.calc_all_performance()
