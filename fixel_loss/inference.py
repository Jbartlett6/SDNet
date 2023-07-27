import torch
import os
import data
import network
import nibabel as nib
import sys
sys.path.append(os.path.join(sys.path[0],'utils'))
import util_funcs

#Defining the appropriate paths:
data_dir = '/media/duanj/F/joe/hcp_2'
save_dir = os.path.join('/home/jxb1336/code/Project_1: HARDI_Recon/FOD-REG_NET/fixel_loss', 'checkpoints')

save_name = 'deep_sh_casc_fix.nii.gz'
subject = '581450'
# fod_path = os.path.join(data_dir, subject, 'T1w', 'Diffusion', 'wmfod.nii.gz')
#fod_path = '/media/duanj/F/joe/hcp_2/581450/T1w/Diffusion/undersampled_fod/gt_fod.nii.gz'
fod_path = '/media/duanj/F/joe/CSD_experiments/deep_sh_casc/inference/581450/inf_fod.nii.gz'
validation_subject_list = ['581450']

experiment_name = 'sh-bignet'

#DEfining the appropriate dataset and dataloader
validation_dataset = data.InferenceFODPatchDataset(data_dir, validation_subject_list, fod_path)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                            shuffle=True, num_workers=4,
                                            drop_last = False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

print('Loading the network and the correct state')
net = network.FixelNet()
net.load_state_dict(torch.load(os.path.join('checkpoints', experiment_name,'model_dict.pt')))
net = net.to(device)
net.eval()

sampling_directions = torch.load(os.path.join('utils/1281_directions.pt'))
order = 8 
P = util_funcs.construct_sh_basis(sampling_directions, order)
P = P.to(device)

#Initialising the output
print('Initialising the output image')
out = torch.zeros((145,174,145)).to(device)

#Performing inference on the input image
with torch.no_grad():
    print('Performing the inference loop')
    for i , data in enumerate(validation_dataloader):
        input_fod, coords = data
        input_fod, coords = input_fod.to(device), coords.to(device)
        
        if i%10000 == 9999:
            print(i, '/', len(validation_dataset))

        
        with torch.no_grad():
            out[coords[:,1], coords[:,2], coords[:,3]] = torch.argmax(net(input_fod[:,:45]),dim=1).float()
            #out[coords[:,1], coords[:,2], coords[:,3]] = torch.argmax(net(torch.matmul(P,input_fod[:,:45].unsqueeze(-1))[:,:,0]),dim=1).float()

print('Finished Inference')

print('Saving the image in nifti format.')
if os.path.isdir(os.path.join(save_dir, str(subject))) == False:
    os.mkdir(os.path.join(save_dir, str(subject)))
x = out.float()
x = x.detach().to('cpu').numpy()
im = nib.Nifti1Image(x, affine=validation_dataset.affine)
nib.save(im, os.path.join(save_dir, str(subject), save_name))
#torch.save(out, os.path.join(save_dir, str(subject), 'inf_fod.pth'))
    