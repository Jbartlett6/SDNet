import torch
import sys
import os
sys.path.append(os.path.join(sys.path[0],'..'))
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys 
import argparse
import data
import argparse

print('Running dwi_undersampled.py:')


parser = argparse.ArgumentParser(description='''Script to undersampled the fully sampled DWI data of a given subject and save it in
                                                the corresponding folder in the data directory''')

parser.add_argument('data_path',type = str, help = 'The location of the hcp data, where the subject folders can be found')
parser.add_argument('subject',type = str, help = 'The subject numbers which will be used to train the network')
parser.add_argument('save_folder',type = str, help = '''The name of the folder to save the undersampled data, bvalues, bvectors and any other files to.
                    This folder will be created in the diffusion folder within the directory corresponding to the subject.''')
parser.add_argument('--dwi_per_shell', type = int, help = 'The number of DWIs in each shell of the undersampled dataset.')
parser.add_argument('--bool_7T', type = bool, help = 'whether the processing is being applied to the 7T data or not.')
args = parser.parse_args()
#usamp = int(sys.argv[1])
#usamp = args.undersampling_rate

#os.mkdir(os.path.join(args.data_path, args.subject,'T1w','Diffusion_7T',args.save_folder))
print('Initialising dataset')
print(args.bool_7T)
d = data.UndersampleDataset(args.subject, args.data_path , undersample_val=args.dwi_per_shell, T7=args.bool_7T, save_folder=args.save_folder)
print('Dataset initialised')
d.all_save()
    