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

def per_subject_inference(subject, opts, data):
    '''
    A function which conducts inference for a single subject according to the options which are supplied to the function. 
    '''
    #Loading the options which will be used in the inference stage.
    print('Loading options')
    opts = options.network_options()
    print(opts.__dict__)

    #Setting the paths to be used.
    print('Setting paths')
    if os.path.isdir(os.path.join('checkpoints',opts.experiment_name, 'inference')) == False:
        os.mkdir(os.path.join('checkpoints',opts.experiment_name, 'inference'))
    save_dir = os.path.join('checkpoints',opts.experiment_name, 'inference')
    model_path = os.path.join('checkpoints', opts.experiment_name, 'models', opts.model_name)
    device = torch.device('cuda')

    #Loading the network
    print('Loading the network and the correct state')
    net = Convcsdcfrnet.CSDNet(opts)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()
    #Initialising the infeence dataset for the given subject. A new dataset will be initialised for each subject in the eval loop.
    print('Initialising the inference dataset and dataloader')
    inf_tmp = [subject]
    dataset =  data.DWIPatchDataset(opts.data_dir, inf_tmp, True, opts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                            shuffle=False, num_workers=8)


    
    #Initialising the output
    print('Initialising the output image')
    #out = F.pad(torch.zeros((145,174,145,47)),(0,0,5,5,5,5,5,5), mode='constant').to(device)
    out = F.pad(torch.zeros((173,207,173,47)),(0,0,5,5,5,5,5,5), mode='constant').to(device)
    
    with torch.no_grad():
        print('Performing the inference loop')
        for i , data in enumerate(dataloader):
            signal_data, _, AQ, coords = data
            signal_data, AQ, coords = signal_data.to(device), AQ.to(device), coords.to(device)
            
            if i%20 == 19:
                print(i*256, '/', len(dataset))

            
            with torch.no_grad():
                out[coords[:,1], coords[:,2], coords[:,3], :] = net(signal_data, AQ).squeeze()
    

    
    
    
    print('Saving the image in nifti format.')
    if os.path.isdir(os.path.join(save_dir, str(subject))) == False:
        os.mkdir(os.path.join(save_dir, str(subject)))
    x = out[5:-5,5:-5,5:-5,:].float()
    x = x.detach().to('cpu').numpy()
    im = nib.Nifti1Image(x, affine=dataset.aff)
    nib.save(im, os.path.join(save_dir, str(subject), 'inf_fod.nii.gz'))
    torch.save(out, os.path.join(save_dir, str(subject), 'inf_fod.pth'))

    os.system('mrconvert' +' '+os.path.join(save_dir, str(subject), 'inf_fod.nii.gz')+ ' ' + '-coord 3 0:44' + ' '+ os.path.join(save_dir, str(subject), 'inf_wm_fod.nii.gz'))

    #os.system('bash /home/jxb1336/code/postproc_funcs/ACC.sh'+ ' ' + str(args.save_path) + ' ' + str(gt_path))