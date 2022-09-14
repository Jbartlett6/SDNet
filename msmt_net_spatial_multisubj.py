from cmath import inf
from operator import not_
import sys
import os 
sys.path.append(os.path.join(sys.path[0],'models'))
sys.path.append(os.path.join(sys.path[0],'utils'))
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np
import util 
import data
import options
#import FODCSDNet as csdnet
import Convcsdnet
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler 
import yaml

if __name__ == '__main__':
    opts = options.network_options()
    print(opts.__dict__)

    #Initalising the tensorboard writer
    plt.switch_backend('agg')

    # d_train = data.DWIPatchDataset(opts.data_dir, opts.train_subject_list, inference=False)
    # d_val = data.DWIPatchDataset(opts.data_dir, opts.val_subject_list, inference=False)

    #Experimental
    d_train = data.ExperimentPatchDataset(opts.data_dir, ['100206'], inference=False)
    d_val = data.ExperimentPatchDataset(opts.data_dir, ['100307'], inference=False)

    train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=opts.batch_size,
                                            shuffle=True, num_workers=opts.train_workers)
    val_dataloader = torch.utils.data.DataLoader(d_val, batch_size=opts.batch_size,
                                            shuffle=True, num_workers=opts.val_workers)


    criterion = torch.nn.MSELoss(reduction='mean')
    
    param_list = [150]
    

    print('Initialising Model')
    net = Convcsdnet.FCNet(opts)
    P = net.P.to(opts.device)
    net = nn.DataParallel(net)
    net = net.to(opts.device)
    #Either loading the existing best model path, or creating the experiment directory depending on the continue training flag.
    model_save_path = os.path.join('checkpoints', opts.experiment_name, 'models')
    plot_offset = 0
    previous_loss = 0
    if opts.continue_training:
        assert os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), 'The experiment ' + opts.experiment_name + ''' does not exist so model parameters cannot be loaded. 
                                                                            Either change continue training flag to create another experiment, or change the experiment name
                                                                            to load an existing experiment'''

        net.load_state_dict(torch.load(os.path.join(model_save_path,'best_model.pth')))
        
        with open(os.path.join(model_save_path,'training_details.yml'), 'r') as file:
            training_details = yaml.load(file, yaml.loader.SafeLoader)
        plot_offset = training_details['plot_step']
        best_loss = training_details['best loss']
        best_val_ACC = training_details['best ACC']
        global_epochs = training_details['epochs_count']
        print('Plot offset is:'+str(plot_offset))
        
        
    else:
        assert not os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), f'The experiment {opts.experiment_name} already exists, please select another experiment name'
        os.mkdir(os.path.join('checkpoints', opts.experiment_name))
        os.mkdir(os.path.join('checkpoints', opts.experiment_name, 'models'))
        best_loss = math.inf
        best_val_ACC = 0
        global_epochs = 0
        
    writer = SummaryWriter(os.path.join('checkpoints', opts.experiment_name,'runs'))
    os.system('tensorboard --port=8008 --logdir'+' '+ os.path.join('checkpoints', opts.experiment_name,'runs')+' &')


    print(net)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'The number of parameters in the model is: {param_num}')
    optimizer = torch.optim.Adam(net.parameters(), lr = opts.lr, betas = (0.9,0.999), eps = 1e-8)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    early_stopping_counter = 0
    previous_loss = inf
    #Running the training loop,in this case for spatial deep reg
    
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        #Breaking out of this loop also if the inner loop has been broken out of due to early stopping.
        running_loss = 0.0
        #0 specifies 
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels, AQ = data
            inputs, labels, AQ = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Training loop: forward + loss calc + backward + optimize
            net.train()
            
            outputs = net(inputs, AQ)
            if opts.loss_type == 'sh':
                mae_criterion = criterion(outputs.squeeze()[:,:45], labels[:,:45])
            elif opts.loss_type == 'sig':
                out_dir = torch.matmul(P,outputs).squeeze()
                gt_dir = torch.matmul(P,labels.unsqueeze(2)).squeeze()
                mae_criterion = criterion(out_dir, gt_dir)       
            
            loss = mae_criterion
            loss.backward()
            optimizer.step()
            
            # Adding the loss calculated for the current minibatch to the running training loss
            running_loss += loss.item()
            
            #Print the training loss and calculate the validation loss every 2000 batches.
            if i % 20 == 19:    
                #Printing the average training loss per batch over the last 2000 batches.
                
                writer.add_scalar('Training Loss', running_loss/20, (len(train_dataloader)*epoch)+i+plot_offset)
                #running_sh_tracker.append(running_sh/100)
                print(f'[{global_epochs+epoch + 1}, {i + 1:5d}] training loss: {running_loss/20:.7f}')
                running_loss = 0.0
                val_loss = 0
                
                # #Calculating the average validation loss over 10 random batches from the validation set.
                with torch.no_grad():
                    #Initialising/resetting the cumulative loss values
                    val_loss = 0.0
                    train_loss = 0.0
                    acc_loss = 0.0
                    non_neg = 0.0
                    val_temp_dataloader = iter(val_dataloader)
                    for j in range(10):
                        data = val_temp_dataloader.next()
                        inputs, labels, AQ = data
                        inputs, labels, AQ = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device)

                        #Could put this in a function connected with the model or alternatively put it in a function on its own
                        net.eval()
                        outputs = net(inputs, AQ)
                        #Records only the white matter FOD spherical harmonic coefficients.
                        loss = criterion(outputs.squeeze()[:,:45], labels[:,:45])
                        val_loss += loss.item()
                        acc_loss += util.ACC(outputs,labels).mean()
                        non_neg += torch.sum((torch.matmul(P, outputs)<-0.01).squeeze(),axis = -1).float().mean()
                
                #Plotting the results using tensorboard.
                writer.add_scalar('Validation Loss', val_loss/10,(len(train_dataloader)*epoch)+i+plot_offset)
                writer.add_scalar('Validation ACC', acc_loss/10 ,(len(train_dataloader)*epoch)+i+plot_offset)
                writer.add_scalar('Validation Non-Negativity Tracker', (non_neg/10).detach().to('cpu').numpy() ,(len(train_dataloader)*epoch)+i+plot_offset)
                writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, (len(train_dataloader)*epoch)+i+plot_offset)
                writer.add_scalar('Non-negativity Regularisation Lambda', net.module.neg_reg, (len(train_dataloader)*epoch)+i+plot_offset)
                writer.add_scalar('Sigmoid Slope', net.module.alpha, (len(train_dataloader)*epoch)+i+plot_offset)

                #Keeping track of the best validation score and implementing early stopping.
                print('Best Loss', best_loss)
                print('Early stopping counter', early_stopping_counter)

                
                if acc_loss/10 > best_val_ACC:
                    best_val_ACC = acc_loss/10
                
                if val_loss/10 < best_loss:
                    
                    best_loss = val_loss/10
                    save_path = os.path.join(model_save_path, 'best_model.pth')
                    torch.save(net.state_dict(), save_path)

                    training_details = {'epochs_count': epoch,
                                        'batch_size':opts.batch_size, 
                                        'minibatch': i, 
                                        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                        'best loss': best_loss,
                                        'best ACC': float(best_val_ACC),
                                        'plot_step':(len(train_dataloader)*epoch)+i+plot_offset,
                                        'deep_reg': float(net.module.deep_reg),
                                        'neg_reg':float(net.module.neg_reg),
                                        'alpha':float(net.module.alpha),
                                        'loss_type':opts.loss_type,
                                        'learn_lambda':opts.learn_lambda,
                                        'Number of Parameters':param_num}

                    with open(os.path.join(model_save_path,'training_details.yml'), 'w') as file:
                        documents = yaml.dump(training_details, file)
                

        #Early stopping implementation.
        current_loss = best_loss
        if current_loss > previous_loss:
            early_stopping_counter = early_stopping_counter+1
        
        if early_stopping_counter > opts.early_stopping_threshold:
                    print(f'Training stopped at epoch {global_epochs+epoch} due to Early stopping and minibatch {i}, the best validation loss achieved is: {best_loss}')
                    break
        
        previous_loss = current_loss

        
    # with open('/media/duanj/F/joe/Project_1_recon/Experiments/csd_net/param_tuning1/results_2.txt', 'a') as txt:
    #     txt.write('\n Deep Regularisation:'+str(parameters['deep reg'])+'Non Negative Regularisation: '+str(parameters['neg reg'])+' Minimum validation loss achieved: '+str(best_loss)+' achieved at batch: '+'Maximum ACC achieved: ' + str(best_val_ACC)+ 'hidden layer width = 256')
            
    print('Finished Training')
