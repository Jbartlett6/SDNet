import sys 
import os
sys.path.append(os.path.join(sys.path[0],'..','models'))
sys.path.append(os.path.join(sys.path[0],'..'))
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

parser = argparse.ArgumentParser(description='Perform inference on using CSDNet on some test subjects and records some metrics relating to the perfromance on these subjects.')

# parser.add_argument('model_path', type=str, help = 'The path at which the model is saved')
# parser.add_argument('save_dir',type=str, help = 'The directory to which to save the final images once inference has been performed.')
# args = parser.parse_args()

device = torch.device('cuda')

parameters = {  'epochs': 1,
                    'lr':1e-5,
                    'deep reg': 6000000/(4000**2),
                    'mag reg': 0, #0.0002*1.7e+9
                    'neg reg': 9000000/(4000**2), #1276**2, #1*(1000000000), #1276**2,
                    'batch size': 512}

inference_subject_list = ['104820']
data_path = '/media/duanj/F/joe/hcp_2'
#save_dir = args.save_dir
save_dir = '/home/jxb1336/code/Project_1: HARDI_Recon/FOD-REG_NET/CSDNet_dir/checkpoints/CSD_main/inference'
#model_path = args.model_path
model_path = '/home/jxb1336/code/Project_1: HARDI_Recon/FOD-REG_NET/CSDNet_dir/checkpoints/CSD_main/models/best_model.pth'
if __name__ == '__main__':
    
    net = csdnet.FCNet(device, parameters['deep reg'], parameters['neg reg'], 150)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load( model_path))
    net = net.to(device)

    for subj in inference_subject_list:
        inf_tmp = [subj]
        print(inf_tmp)
        dataset =  data.DWIPatchDataset(data_path, inf_tmp, True)
        print(dataset.__len__())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                                shuffle=True, num_workers=8)

        # gt_path = '/media/duanj/F/joe/hcp_2/'+str(subj)+'/T1w/Diffusion/undersampled_fod/gt_wm_fod.mif'
        print('out')
        out = torch.zeros((145,174,145,47)).to(device)
        print('2')

        for i , data in enumerate(dataloader):
            print('1')
            print(i)
            signal_data, _, AQ, coords = data
            signal_data, AQ, coords = signal_data.to(device), AQ.to(device), coords.to(device)

            if i%20 == 19:
                print(i*256, '/', len(dataset))

            
            with torch.no_grad():
                out[coords[:,1], coords[:,2], coords[:,3], :] = net(signal_data, AQ).squeeze()

        x = out.float()
        #x = x[:,:,70,:]
        x = x.detach().to('cpu').numpy()
        print(x.shape)
        im = nib.Nifti1Image(x, affine=dataset.aff , header=dataset.head)
        nib.save(im, save_dir+'/'+str(subj)+'_inference.nii.gz')

        #os.system('bash /home/jxb1336/code/postproc_funcs/ACC.sh'+ ' ' + str(args.save_path) + ' ' + str(gt_path))