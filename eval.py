import sys
import os 
import argparse
import numpy as np

def mrstats_interpreter(path):
    with open(path, 'r') as f:
        stats_txt = f.read()

    mean_list = [float(x.split(' ')[12]) for x in stats_txt.split('\n')[2::3]]
    return np.mean(mean_list)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Perform a training loop for the ')
    parser.add_argument('experiment_name', type=str, help = 'The experiment name in the checkpoint directory')
    parser.add_argument('dataset_dir', type=str, help = 'The location of the hcp data directory')
    args = parser.parse_args()

    file_dir = args.experiment_name
    inference_path = os.path.join('.','checkpoints',file_dir,'inference')
    subjects = os.listdir(inference_path)
    #Include all measures of accuracy in this loop and save them to the appropriate destination
    for subj in subjects:
        print('Performing accuracy measures for subject: '+ subj)
        
        print('Calculating ACC')
        gt_fod_path = os.path.join(args.dataset_dir, subj, 'T1w', 'Diffusion', 'undersampled_fod', 'gt_wm_fod.nii.gz')
        inf_wm_path = os.path.join(inference_path, subj, 'inf_wm_fod.nii.gz')
        save_path = os.path.join(inference_path, subj)
        os.system('bash utils/ACC.sh ' + gt_fod_path + ' ' + inf_wm_path +' '+save_path+' '+subj) 
        os.system

        print('Performing fixel based analysis')
        os.system('bash utils/FBA.sh ' + inf_wm_path + ' ' + subj +' '+save_path)

    with open(os.path.join(inference_path, 'all_stats.txt'), 'a') as f:
            #Writing the average stats for the ACC into a text file.
            f.write('The mean ACC over the white matter is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wm_acc_stats.txt'))) + '\n')
            f.write('The mean ACC over the whole brain is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wb_acc_stats.txt'))) + '\n')
            
            #Writing the average stats for the pae into a text file
            f.write('The mean pae over the white matter is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wm_pae_stats.txt'))) + '\n')
            f.write('The mean pae over the whole brain is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wb_pae_stats.txt'))) + '\n')
            

            #Writing the average stats for the afde into a text file. 
            f.write('The mean afde in the white matter is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wm_afde_stats.txt'))) + '\n')
            f.write('The mean afde in the whole brain is: \n')
            f.write(str(mrstats_interpreter(os.path.join(inference_path, 'wb_afde_stats.txt'))) + '\n')