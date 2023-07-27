
from genericpath import isdir
import sys 
import os
sys.path.append(os.path.join(sys.path[0],'models'))
# sys.path.append(os.path.join(sys.path[0],))
sys.path.append(os.path.join(sys.path[0],'utils'))
import options
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.special
import nibabel as nib
import random 
import matplotlib.pyplot as plt 
import time
import numpy as np
import argparse
import csdnet
import data
import Convcsdnet
import Convcsdcfrnet

class InferenceClass():
    def __init__(self, model_name, experiment_name, opts):
        # self.experiment_name = experiment_name
        # self.model_name = model_name
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opts = opts

    def set_paths(self):
        #Setting the paths to be used.
        print('Setting paths')

        #Creating the inference directory to save the inferred FODs
        if os.path.isdir(os.path.join('checkpoints',self.opts.experiment_name, 'inference')) == False:
            os.mkdir(os.path.join('checkpoints',self.opts.experiment_name, 'inference'))
        self.save_dir = os.path.join('checkpoints',self.opts.experiment_name, 'inference')

        #Setting the model path name (where the weights of the model are saved)
        self.model_path = os.path.join('checkpoints', self.opts.experiment_name, 'models', self.opts.model_name)

    def print_paths(self):
        print(self.save_dir)
        print(self.model_path)
        
    def load_options(self):
        print('Loading options')
        self.opts = options.network_options()
        print(self.opts.__dict__)

    def load_network(self):
        #Loading the network
        print('Loading the network and the correct state')
        net = Convcsdcfrnet.CSDNet(self.opts)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(self.model_path))

        net = net.to(self.opts.device)
        self.net = net.eval()

    def load_data(self,subject):
        print('Initialising the inference dataset and dataloader')
        inf_tmp = [subject]
        
        dataset =  data.DWIPatchDataset(inf_tmp, True, False, self.opts)
        self.dataset_length = len(dataset)
        self.dataset_affine = dataset.aff

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                            shuffle=False, num_workers=8)
        
    def perform_inference(self):
        #Initialising the output
        print('Initialising the output image')
        out = F.pad(torch.zeros((145,174,145,47)),(0,0,5,5,5,5,5,5), mode='constant').to(self.opts.device)
        
        
        with torch.no_grad():
            print('Performing the inference loop')
            for i , data in enumerate(self.dataloader):
                signal_data, _, AQ, _,coords = data
                signal_data, AQ, coords = signal_data.to(self.opts.device), AQ.to(self.opts.device), coords.to(self.opts.device)
                
                if i%20 == 19:
                    print(i*256, '/', self.dataset_length)

                out[coords[:,1], coords[:,2], coords[:,3], :] = self.net(signal_data, AQ).squeeze()

        self.FOD = out

    def save_FOD(self, subject):
        print('Saving the image in nifti format.')
        if os.path.isdir(os.path.join(self.save_dir, str(subject))) == False:
            os.mkdir(os.path.join(self.save_dir, str(subject)))
        x = self.FOD[5:-5,5:-5,5:-5,:].float()
        x = x.detach().to('cpu').numpy()
        im = nib.Nifti1Image(x, affine=self.dataset_affine)
        nib.save(im, os.path.join(self.save_dir, str(subject), 'inf_fod.nii.gz'))
        
        os.system(f'mrconvert {os.path.join(self.save_dir, str(subject), "inf_fod.nii.gz")} -coord 3 0:44 {os.path.join(self.save_dir, str(subject), "inf_wm_fod.nii.gz")}')

    def run_seq(self, subject):
        self.set_paths()
        self.print_paths()
        self.load_options()
        self.load_network()
        self.load_data(subject)
        self.perform_inference()
        self.save_FOD(subject)

