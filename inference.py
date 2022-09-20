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

print('Loading options')
opts = options.network_options()
print(opts.__dict__)
print('arguments loaded')

print('Setting paths')
if os.path.isdir(os.path.join('checkpoints',opts.experiment_name, 'inference')) == False:
    os.mkdir(os.path.join('checkpoints',opts.experiment_name, 'inference'))


save_dir = os.path.join('checkpoints',opts.experiment_name, 'inference')
model_path = os.path.join('checkpoints', opts.experiment_name, 'models', opts.model_name)

device = torch.device('cuda')
print('Completed setting paths')



if __name__ == '__main__':
    net = Convcsdnet.FCNet(opts)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    print('Initialising the inference dataset and dataloader')
    inf_tmp = [opts.subject]
    dataset =  data.DWIPatchDataset(opts.data_path, inf_tmp, True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                            shuffle=True, num_workers=8)

    
    print('Initialising the output image')
    out = torch.zeros((145,174,145,47)).to(device)

    print('Performing the inference loop')
    for i , data in enumerate(dataloader):
        signal_data, _, AQ, coords = data
        signal_data, AQ, coords = signal_data.to(device), AQ.to(device), coords.to(device)

        if i%20 == 19:
            print(i*256, '/', len(dataset))

        
        with torch.no_grad():
            out[coords[:,1], coords[:,2], coords[:,3], :] = net(signal_data, AQ).squeeze()

    print('Saving the image in nifti format.')
    x = out.float()
    #x = x[:,:,70,:]
    x = x.detach().to('cpu').numpy()
    print(x.shape)
    im = nib.Nifti1Image(x, affine=dataset.aff , header=dataset.head)
    nib.save(im, save_dir+'/'+str(opts.subject)+'_inference.nii.gz')

    #os.system('bash /home/jxb1336/code/postproc_funcs/ACC.sh'+ ' ' + str(args.save_path) + ' ' + str(gt_path))