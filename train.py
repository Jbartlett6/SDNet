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
import Convcsdcfrnet
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler 
import yaml
import tracker 

if __name__ == '__main__':
    opts = options.network_options()

    #Initalising the tensorboard writer
    plt.switch_backend('agg')
    
    train_dataloader, val_dataloader = data.init_dataloaders(opts)
    criterion = torch.nn.MSELoss(reduction='mean')
    net, P, param_num, current_training_details, model_save_path = Convcsdcfrnet.init_network(opts)
    optimizer = torch.optim.Adam(net.parameters(), lr = opts.lr, betas = (0.9,0.999), eps = 1e-8)
    loss_tracker = tracker.LossTracker(P,criterion)    
    visualiser = tracker.Vis(opts, train_dataloader)
    writer = SummaryWriter(os.path.join('checkpoints', opts.experiment_name,'runs'))

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    early_stopping_counter = 0
    
    #Running the training loop,in this case for spatial deep reg
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels, AQ = data
            inputs, labels, AQ = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device)
            
            # zero the parameter gradients and setting network to train
            optimizer.zero_grad()
            net.train()
            
            #The feeding the data forward through the network.
            outputs = net(inputs, AQ)
            
            #Calculating the loss function, backpropagation and stepping the optimizer
            loss = criterion(outputs.squeeze()[:,:45], labels[:,:45])
            loss.backward()
            optimizer.step()
            
            # Adding the loss calculated for the current minibatch to the running training loss
            loss_tracker.add_running_loss(loss)

            if i%20 == 19:    
                # #Calculating the average validation loss over 10 random batches from the validation set.
                with torch.no_grad():
                    #Forward pass for calculating validation loss
                    val_temp_dataloader = iter(val_dataloader)
                    for j in range(10):
                        data = val_temp_dataloader.next()
                        inputs, labels, AQ = data
                        inputs, labels, AQ = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device)

                        #Could put this in a function connected with the model or alternatively put it in a function on its own
                        net.eval()
                        outputs = net(inputs, AQ)
                        loss_tracker.add_val_losses(outputs,labels)

                #Plotting the results using tensorboard using the visualiser class.
                visualiser.add_scalars(loss_tracker.loss_dict, net, current_training_details, i, epoch)

                #Printing the current best validation loss, and the early stopping counter
                print('Best Loss', current_training_details['best_loss'])
                print('Early stopping counter', early_stopping_counter)

                #Updating the training details.
                current_training_details = tracker.update_details(loss_tracker.loss_dict, current_training_details, model_save_path,
                                                            net, epoch, i, opts, optimizer, param_num, train_dataloader)        
                
                #Resetting the losses for the next set of minibatches
                loss_tracker.reset_losses()
        

        #Early stopping implementation (over epochs).
        current_loss = current_training_details['best_loss']
        if current_loss > current_training_details['previous_loss']:
            early_stopping_counter = early_stopping_counter+1
        
        if early_stopping_counter > opts.early_stopping_threshold:
                    print(f'Training stopped at epoch {current_training_details["global_epochs"]+epoch} due to Early stopping and minibatch {i}, the best validation loss achieved is: {current_training_details["best_loss"]}')
                    break
        
        current_training_details['previous_loss'] = current_loss
        
                
    print('Finished Training')
    
