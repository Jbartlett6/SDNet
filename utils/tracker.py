from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import nibabel as nib
import os
import util
import torch
import yaml


class Vis():
    def __init__(self, opts, train_dataloader):
        self.writer = SummaryWriter(os.path.join('checkpoints', opts.experiment_name,'runs'))
        self.dataloader_length = len(train_dataloader)
        
    def add_scalars(self,losses,net,current_training_details,i,epoch):
        #Training loss and its decomposition :
        self.writer.add_scalar('Training Loss', losses['running_loss']/20, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('FOD Loss', losses['fod_loss']/20, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('Fixel Loss', losses['fixel_loss']/20, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('Fixel Accuracy', losses['fixel_accuracy']/20, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        
        #The validation losses
        self.writer.add_scalar('Validation Loss', losses['val_loss']/10,(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('Validation ACC', losses['acc_loss']/10 ,(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])        
        self.writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])

        print(f'[{current_training_details["global_epochs"]+epoch + 1}, {i + 1:5d}] training loss: {losses["running_loss"]/20:.7f} training fod loss {losses["fod_loss"]/20:.7f}')
        


    def add_fixel_scalars(self,afde_mean, afde_median, pae_mean, pae_median,current_training_details,i,epoch):
        self.writer.add_scalar('AFDE Mean', float(afde_mean), (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('AFDE Median', float(afde_median),(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('PAE Mean', float(pae_mean),(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('PAE Median', float(pae_median),(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
    
    
        

        

class LossTracker():
    def __init__(self,P,criterion):
        self.loss_dict = {'running_loss':0.0, 'val_loss':0.0, 'acc_loss':0.0, 'non_neg':0.0, 'fod_loss':0.0, 'fixel_loss':0.0, 'fixel_accuracy':0.0}
        self.P = P
        self.criterion=criterion

    def reset_losses(self):
        self.loss_dict['val_loss'] = 0.0
        self.loss_dict['acc_loss'] = 0.0
        self.loss_dict['non_neg'] = 0.0
        self.loss_dict['running_loss'] = 0.0
        self.loss_dict['fod_loss'] = 0.0
        self.loss_dict['fixel_loss'] = 0.0
        self.loss_dict['fixel_accuracy'] = 0.0
        
    
    def add_val_losses(self, outputs, labels):
        #loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
        loss = self.criterion(outputs.squeeze(), labels)
        self.loss_dict['val_loss'] += loss.item()
        self.loss_dict['acc_loss'] += util.ACC(outputs,labels).mean()
        self.loss_dict['non_neg'] += torch.sum((torch.matmul(self.P, outputs)<-0.01).squeeze(),axis = -1).float().mean()

    def add_running_loss(self,loss,fod_loss,fixel_loss, fixel_accuracy):
        self.loss_dict['running_loss'] += loss.item()
        self.loss_dict['fod_loss'] += fod_loss.item()
        self.loss_dict['fixel_loss'] += fixel_loss.item()
        self.loss_dict['fixel_accuracy'] += fixel_accuracy.item()



def update_details(losses, current_training_details, model_save_path, net, epoch, i, opts, optimizer, param_num, train_dataloader):
    if losses['acc_loss']/250 > current_training_details['best_val_ACC']:
        current_training_details['best_val_ACC'] = losses['acc_loss']/250

    if losses['val_loss']/250 < current_training_details['best_loss']:
                    
        current_training_details['best_loss'] = losses['val_loss']/250
        save_path = os.path.join(model_save_path, 'best_model.pth')
        torch.save(net.state_dict(), save_path)

        training_details = {'epochs_count': epoch,
                            'batch_size':opts.batch_size, 
                            'minibatch': i, 
                            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                            'best loss': current_training_details['best_loss'],
                            'best ACC': float(current_training_details['best_val_ACC']),
                            'plot_step':(len(train_dataloader)*epoch)+i+current_training_details['plot_offset'],
                            'deep_reg': float(net.module.deep_reg),
                            'neg_reg':float(net.module.neg_reg),
                            'alpha':float(net.module.alpha),
                            'loss_type':opts.loss_type,
                            'learn_lambda':opts.learn_lambda,
                            'Number of Parameters':param_num,
                            'dataset_type':opts.dataset_type}

        with open(os.path.join(model_save_path,'training_details.yml'), 'w') as file:
            documents = yaml.dump(training_details, file)
    
    #Keeping track of the best validation score and implementing early stopping.
        
        

    return current_training_details

def fba_eval(val_dataloader, net,opts, val_affine):
    
    print('Initialising the output image')
    out = F.pad(torch.zeros((62,70,80,47)),(0,0,5,5,5,5,5,5), mode='constant').to(opts.device)
    
    #Need to perform the inference loop over the entire dataloader - to calculate the FODs.
   
    with torch.no_grad():
        net = net.eval()
        print('Performing the inference loop')
        for i, data in enumerate(val_dataloader):
            signal_data, _, AQ, coords = data
            signal_data, AQ, coords = signal_data.to(opts.device), AQ.to(opts.device), coords.to(opts.device)
            
            if i%20 == 19:
                print(i*256, '/', len(val_dataloader)*256)
        
            out[coords[:,1], coords[:,2], coords[:,3], :] = net(signal_data, AQ).squeeze()
        net.train()

    #Inference Paths
    temp_path = os.path.join('checkpoints', opts.experiment_name, 'val_temp')
    #temp_path = os.path.join(opts.data_dir, '100307','T1w','Diffusion','val_temp')
    os.mkdir(temp_path)

    fod_path = os.path.join(temp_path, 'fod.nii.gz')
    fixel_dir = os.path.join(temp_path,'fix_dir')

    afd_path = os.path.join(fixel_dir,'afd.nii.gz')
    afd_im_path = os.path.join(fixel_dir, 'afd_im.nii.gz')

    pa_path = os.path.join(fixel_dir,'pa.nii.gz')
    pa_im_path = os.path.join(fixel_dir,'pa_im.nii.gz')

    #tmps
    abs_afde_path = os.path.join(fixel_dir,'abs_afde.nii.gz')
    tot_afde_path = os.path.join(fixel_dir,'afde.nii.gz')

    abs_pae_path = os.path.join(fixel_dir,'abs_pae.nii.gz')
    tot_pae_path = os.path.join(fixel_dir,'pae.nii.gz')
    stat_path = os.path.join(temp_path,'stat_path.txt')

    #Ground Truth Paths
    gt_fix_dir = os.path.join(opts.data_dir, '100307','T1w','Diffusion','bench_val_fixel_dir')
    gt_afd_im_path = os.path.join(gt_fix_dir, 'afd_im.nii.gz') 
    gt_pa_im_path = os.path.join(gt_fix_dir, 'pa_im.nii.gz') 
    cropped_wm_path = os.path.join(opts.data_dir, '100307','T1w','Diffusion', 'cropped_wm_mask.nii.gz')


    
    out = out[5:-5,5:-5,5:-5,:45].detach().to('cpu').numpy()
    validation_fod = nib.Nifti1Image(out, affine=val_affine)
    nib.save(validation_fod, fod_path)

    
    os.system('fod2fixel -afd afd.nii.gz -peak_amp pa.nii.gz'+ ' ' + str(fod_path) + ' ' + str(fixel_dir))

    #Apply mrtrix to calculate the fixels of the fod
    #calculate the afde_image
    os.system('fixel2voxel -number 11 ' + str(afd_path) + ' none ' + str(afd_im_path))
    os.system('fixel2voxel -number 11 ' + str(pa_path) + ' none ' + str(pa_im_path))

    #Calculate the difference metrics and appropraite stats using mrtrix
    os.system('mrcalc ' + str(afd_im_path) + ' ' + str(gt_afd_im_path) + ' -sub -abs ' + str(abs_afde_path))
    os.system('mrmath ' + str(abs_afde_path) + ' sum ' + str(tot_afde_path) + ' -axis 3')
    
    os.system('mrcalc ' + str(pa_im_path) + ' ' + str(gt_pa_im_path) + ' -sub -abs ' + str(abs_pae_path)) 
    os.system('mrmath ' + str(abs_pae_path) + ' sum ' + str(tot_pae_path) + ' -axis 3')

    os.system('mrstats -mask ' + str(cropped_wm_path) + ' ' + str(tot_afde_path) + ' >> ' + str(stat_path))
    os.system('mrstats -mask ' + str(cropped_wm_path) + ' ' + str(tot_pae_path) + ' >> ' + str(stat_path))

    #Read the stats from the text file into pthon
    means = stat_extract(stat_path,'mean')
    afde_mean = means[0]
    pae_mean = means[1]

    medians = stat_extract(stat_path,'median')
    afde_median = medians[0]
    pae_median = medians[1]

    os.system('rm -r ' + str(temp_path))
    print(afde_mean, afde_median, pae_mean, pae_median)
    
    return afde_mean, afde_median, pae_mean, pae_median
        

def stat_extract(path,stat_name):
    indicies = {'mean':3, 'median': 4, 'std': 5, 'min':6, 'max':7, 'count':8}
    
    with open(path, 'r') as f:
        x = f.read()
    
    y = x.split('\n')
    stats = [y[i] for i in range(len(y)) if i % 2 == 1]
    stat_list = [[i for i in line.split(' ') if i != '' ][indicies[stat_name]] for line in stats]
    return stat_list   
def fixel_accuracy(fix_est, gt_fixel):
    fixel_preds = torch.argmax(fix_est, dim = 1)
    val_acc = torch.sum(fixel_preds == gt_fixel)/gt_fixel.shape[0]
    
    return val_acc

    
