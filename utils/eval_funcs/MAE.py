from tabnanny import check
import numpy as np
import nibabel as nib
import argparse 
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a training loop for the ')
    
    parser.add_argument('--inference_image',type=str, help = 'The path to the inference image')
    parser.add_argument('--subject',type=str, help = 'The subject number')
    parser.add_argument('--save_directory',type=str, help = 'The path to the inference image')
    parser.add_argument('--data_path',type=str, help = 'The path to the data')

    args = parser.parse_args()

    nft_im = nib.load(args.inference_image)
    nft_gt = nib.load(os.path.join(args.data_path, args.subject, 'T1w', 'Diffusion', 'fixel_directory', 'fixel_directions.nii.gz'))

    im_data = nft_im.get_fdata()
    gt_data = nft_gt.get_fdata()

    inter = np.zeros((im_data.shape[0],im_data.shape[1],im_data.shape[2],im_data.shape[3]))
    print(im_data.shape)
    for i in range(im_data.shape[3]):
        inter[:,:,:,i] = im_data[:,:,:,i]*gt_data[:,:,:,i] 

    out = np.zeros((im_data.shape[0],im_data.shape[1],im_data.shape[2],int(im_data.shape[3]/3)))
    #inter = np.zeros((im_data.shape[0],im_data.shape[1],im_data.shape[2],int(im_data.shape[3])))
    for i in range(int(im_data.shape[3]/3)):
        out[:,:,:,i] = np.sum(inter[:,:,:,3*i:3*i+3],axis=3)
    nft_out = nib.Nifti1Image(out, affine = nft_im.affine, header = nft_im.header)
    nib.save(nft_out, 'mae_tmp.nii.gz')
