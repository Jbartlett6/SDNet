'''
Script to perform inference given the the options selected in options.py
'''

import utils.data as data
from models import Convcsdcfrnet
import options

import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm


class InferenceClass():
    '''
    Description:
                Given a list of subsect, as found in opts.test_subject_list, and other model configurations found in opts,
                this class will perform inference on the list of subjects and save them in the in the experiment's folder 
                in the inference folder. The model weights that are loaded are dictatated by the experiment_name and the 
                model_name which indicate the training run and which model instance respectively.
                 

    Methods:
                __init__            - Initialising the paths where the FODs inferred FODs will be save and the network
                                    which is being used to predict the FODs
                set_paths           - Setting the appropriate paths, namely model_path and save_path
                load_network        - Loading the network into the class.
                load_data           - Creating a dataloader for the current_training subject. 
                perform_inference   - Performing inference for the current subject (from the dataloader produced by load_data)
                save_FOD            - Saving the FOD which is currently saved in self.FOD into nifti format.
                inference_loop      - Performing the inference loop for all subjects specfied in the test_subject_list in options.py

    '''
    def __init__(self, opts):
        self.opts = opts
        
        self.device = opts.device
        self.subject_list = opts.test_subject_list
        self.experiment_name = opts.experiment_name
        self.model_name = opts.model_name
        

        self.current_subject = None
        
        self.save_dir, self.model_path = self.set_paths()
        self.net = self.load_network()

    def set_paths(self):
        print('Setting paths')

        # Directory the inferred FODs are to be saved
        save_dir = os.path.join('checkpoints', self.experiment_name, 'inference')
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        # Path where the model weights are loaded from
        model_path = os.path.join('checkpoints', self.experiment_name, 'models', self.model_name)

        return save_dir, model_path
        
    def load_network(self):
        #Loading the network
        print('Loading the network and the correct state')
        net = Convcsdcfrnet.CSDNet(self.opts)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(self.model_path))

        net = net.to(self.device)
        net = net.eval()

        return net

    def perform_inference(self, dataloader):
        #Initialising the output
        print('Initialising the output image')
        out = F.pad(torch.zeros((145,174,145,47)),(0,0,5,5,5,5,5,5), mode='constant').to(self.device)
        
        
        with torch.no_grad():
            print(f'Performing the inference loop for subject {self.current_subject}')
            for i , data in enumerate(tqdm(dataloader)):
                signal_data, _, AQ, _,coords = data
                signal_data, AQ, coords = signal_data.to(self.device), AQ.to(self.device), coords.to(self.device)

                out[coords[:,1], coords[:,2], coords[:,3], :] = self.net(signal_data, AQ).squeeze()

        return out
    
    def save_FOD(self, FOD, dataset_affine):
        print('Saving the image in nifti format.')
        if os.path.isdir(os.path.join(self.save_dir, str(self.current_subject))) == False:
            os.mkdir(os.path.join(self.save_dir, str(self.current_subject)))
        x = FOD[5:-5,5:-5,5:-5,:].float()
        x = x.detach().to('cpu').numpy()
        im = nib.Nifti1Image(x, affine=dataset_affine)
        nib.save(im, os.path.join(self.save_dir, str(self.current_subject), 'inf_fod.nii.gz'))
        
        os.system(f'mrconvert {os.path.join(self.save_dir, str(self.current_subject), "inf_fod.nii.gz")} -coord 3 0:44 {os.path.join(self.save_dir, str(self.current_subject), "inf_wm_fod.nii.gz")}')

    def inference_loop(self):
        for subject in self.subject_list:
            self.current_subject = subject
            dataset_affine, dataloader = load_data(subject, self.opts)
            FOD = self.perform_inference(dataloader)
            self.save_FOD(self.save_dir, subject, FOD, dataset_affine)


def load_data(current_subject, opts):
    print(f'Initialising the inference dataset and dataloader for subject {current_subject}')
    inf_tmp = [current_subject]
    
    dataset =  data.DWIPatchDataset(inf_tmp, True, False, opts)
    
    dataset_affine = dataset.aff

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                        shuffle=False, num_workers=8)
    
    return dataset_affine, dataloader


if __name__ == '__main__':
    opts = options.NetworkOptions()
    inf_obj = InferenceClass(opts)
    inf_obj.inference_loop()