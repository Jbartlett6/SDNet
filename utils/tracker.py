from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import nibabel as nib
import os
import util
import torch
import yaml
import sys
import numpy as np
import time

class Vis():
    '''
    Description: 
                Class for visualising the training curves in tensorboard.
    Methods:
                __init__    - Initialising the Summary writer and losses to be tracked
                add_scalars - Adding the scalar to be tracked to the trensorboard writer.
    '''
    def __init__(self, opts, train_dataloader):
        self.opts = opts
        self.writer = SummaryWriter(os.path.join('checkpoints', opts.experiment_name,'runs'))
        self.dataloader_length = len(train_dataloader)
        self.train_losses = ['Training Loss', 'FOD Loss', 'Fixel Loss', 'Fixel Accuracy']
        self.val_losses = ['Validation Loss', 'Validation ACC', 'Validation Fixel Loss', 'Validation Fixel Accuracy']
        
    def add_scalars(self, train_losses, val_losses, current_training_details, i, epoch):
        step = (self.dataloader_length*epoch)+i+current_training_details['plot_offset']
        # step = (self.dataloader_length*epoch)+i

        #Training loss and its decomposition :
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(loss_name, loss_value/self.opts.val_freq, step)

        #The validation losses
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(loss_name, loss_value/self.opts.val_iters, step)
 
        # #self.writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, step)
        print(f'[{current_training_details["global_epochs"]+epoch + 1}, {i + 1:5d}] training loss: {train_losses["Training Loss"]/self.opts.val_freq:.7f} training fod loss {train_losses["FOD Loss"]/self.opts.val_freq:.7f}')
        

class LossTracker():
    def __init__(self,criterion):
        
        self.train_losses = ['Training Loss', 'FOD Loss', 'Fixel Loss', 'Fixel Accuracy']
        self.val_losses = ['Validation Loss', 'Validation ACC', 'Validation Fixel Loss', 'Validation Fixel Accuracy']
        
        self.train_loss_dict = dict.fromkeys(self.train_losses, 0.0)
        self.val_loss_dict = dict.fromkeys(self.val_losses, 0.0)
        
        self.criterion=criterion

    def reset_losses(self):
        self.train_loss_dict = dict.fromkeys(self.train_losses, 0.0)
        self.val_loss_dict = dict.fromkeys(self.val_losses, 0.0)

    
    def add_val_losses(self, outputs, labels, val_fixel_loss, val_fixel_accuracy):
        
        loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
        
        self.val_loss_dict['Validation Loss'] += loss.item()
        self.val_loss_dict['Validation ACC'] += util.ACC(outputs,labels).mean()
        self.val_loss_dict['Validation Fixel Loss'] += val_fixel_loss.item()
        self.val_loss_dict['Validation Fixel Accuracy'] += val_fixel_accuracy.item()

    def add_running_loss(self,loss,fod_loss,fixel_loss, fixel_accuracy):
        
        self.train_loss_dict['Training Loss'] += loss.item()
        self.train_loss_dict['FOD Loss'] += fod_loss.item()
        self.train_loss_dict['Fixel Loss'] += fixel_loss.item()
        self.train_loss_dict['Fixel Accuracy'] += fixel_accuracy.item()
        



def update_details(train_losses, val_losses, current_training_details, model_save_path, net, epoch, i, opts, optimizer, param_num, train_dataloader):
    if val_losses['Validation ACC']/opts.val_iters > current_training_details['best_val_ACC']:
        current_training_details['best_val_ACC'] = val_losses['Validation ACC']/opts.val_iters

    if val_losses['Validation Loss']/opts.val_iters < current_training_details['best_loss']:
                    
        current_training_details['best_loss'] = val_losses['Validation Loss']/opts.val_iters
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
                            'learn_lambda':opts.learn_lambda,
                            'Number of Parameters':param_num}
        
        training_details_string = [f'{name}: {value} \n' for name, value in training_details.items()]

        train_log_path = os.path.join('checkpoints', opts.experiment_name, 'logs', 'training.log')
        with open(train_log_path, 'w') as trainlog:
            [trainlog.write(entry) for entry in training_details_string]

        with open(os.path.join(model_save_path,'training_details.yml'), 'w') as file:
            documents = yaml.dump(training_details, file)

    return current_training_details



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


class EarlyStopping():
    def __init__(self):
        self.early_stopping_counter = 0

    def early_stopping_update(self,current_training_details,opts,epoch,i):
        current_loss = current_training_details['best_loss']
        if current_loss > current_training_details['previous_loss']:
            self.early_stopping_counter = self.early_stopping_counter+1

        if self.early_stopping_counter > opts.early_stopping_threshold:
            print(f'Training stopped at epoch {current_training_details["global_epochs"]+epoch} due to Early stopping and minibatch {i}, the best validation loss achieved is: {current_training_details["best_loss"]}')
            sys.exit()

        return current_loss

class RuntimeTracker():
    def __init__(self, runtime_memory, runtime_log_path, opts, iterations_per_epoch):
        self.opts = opts
        self.runtimes = {}
        self.starttimes = {}
        self.timerbool = {}

        self.runtime_memory = runtime_memory
        self.runtime_log_path = runtime_log_path
        self.iterations_per_epoch = iterations_per_epoch

        self.add_runtime_tracker('training iter')
    
    def add_runtime_tracker(self, name):
        self.runtimes[name] = [0 for i in range(self.runtime_memory)]
        self.starttimes[name] = 0
        self.timerbool[name] = False

    def start_timer(self, name):
        #assert self.timerbool{name} == False:
        self.timerbool[name] = True
        self.starttimes[name] = time.time()

    def stop_timer(self, name):
        self.timerbool[name] = False

        new_runtimes = self.runtimes[name][1:]
        new_runtimes.append(time.time() - self.starttimes[name])
        self.runtimes[name] = new_runtimes

    def write_runtimes(self):
        
        single_epoch_time = (np.mean(self.runtimes['training iter'])*self.iterations_per_epoch)/3600
        
        string_list = [f'Average runtime for {name} is {np.mean(value)} seconds \n' for name, value in self.runtimes.items()]
        with open(self.runtime_log_path, 'w') as rtlog:
            [rtlog.write(entry) for entry in string_list]
            rtlog.write('\n')
            rtlog.write(f'Approximate time to run 1 epoch (excluding validation loop): {round(single_epoch_time,2)} hours \n')
            rtlog.write(f'Time to run all {self.opts.epochs} epochs: {round(single_epoch_time*self.opts.epochs, 2)} hours')

    

    
