import utils.util as util

import time
import yaml
import os

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


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
        
    def add_scalars(self, train_losses, val_losses, epoch, iterations):
        step = iterations

        #Training loss and its decomposition :
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(loss_name, loss_value/self.opts.val_freq, step)

        #The validation losses
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(loss_name, loss_value/self.opts.val_iters, step)
 
        # #self.writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, step)
        print(f'[{epoch + 1}, {iterations} total iterations] training loss: {train_losses["Training Loss"]/self.opts.val_freq:.7f} training fod loss {train_losses["FOD Loss"]/self.opts.val_freq:.7f}')
        

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
        
class TrainingLogger():
    
    def __init__(self, model_save_path, param_num, opts):
        self.model_save_path = model_save_path
        self.param_num = param_num
        self.opts = opts
    
    def update_training_logs(self, train_losses, val_losses, current_training_details, 
                            net, epoch, optimizer, es, iterations):
        
        if val_losses['Validation ACC']/self.opts.val_iters > current_training_details['best_val_ACC']:

            current_training_details['best_val_ACC'] = val_losses['Validation ACC']/self.opts.val_iters

        if val_losses['Validation Loss']/self.opts.val_iters < current_training_details['best_loss']:
                        
            current_training_details['best_loss'] = val_losses['Validation Loss']/self.opts.val_iters
            
            training_state_dict = {'net_state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
            'earlystopping_state': es.state_dict(),
            'epochs': epoch,
            'iterations': iterations,
            'opts': self.opts,
            'current_training_details': current_training_details}
                    
            torch.save(training_state_dict, os.path.join(self.model_save_path, 'best_training.pth'))


        training_state = {'epochs_count': epoch, # Training state 
                            'iterations': iterations, # Training state
                            'lr': optimizer.state_dict()['param_groups'][0]['lr'], # Training state
                            'deep_reg': float(net.module.deep_reg), # Training state
                            'neg_reg':float(net.module.neg_reg), # Training state
                            'alpha':float(net.module.alpha),
                            'Number of Parameters':self.param_num} # Training state

        stopping_criteria = {'epochs_count': epoch,
                            'epoch_limit': self.opts.epochs,
                            'iterations': iterations,
                            'iterations limit': self.opts.iteration_limit,
                            'early stopping counter': es.early_stopping_counter,
                            'early stopping threshold': self.opts.early_stopping_threshold,
                            'early stopping best loss': es.best_loss,
                            'early stopping best iteration': es.best_loss_iter,
                            'early stopping highest count': es.highest_counter,
                            'early stopping lr sched count': es.lr_scheduler_count  
                            }

        config = self.opts.__dict__

        training_performance = train_losses

        performance = {'best loss': current_training_details['best_loss'], # Performance measure
                    'best ACC': float(current_training_details['best_val_ACC'])} # Performance measure }

        training_details = {**training_state, **stopping_criteria, **config, **performance}
                        
        with open(os.path.join(self.model_save_path,'training_details.yml'), 'w') as file:
            yaml.dump(training_details, file)
            

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





class RuntimeTracker():
    def __init__(self, runtime_memory, runtime_log_path, opts, iterations_per_epoch):
        self.opts = opts
        self.runtimes = {}
        self.starttimes = {}
        self.timerbool = {}

        self.init_time = time.time()
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
        current_time = time.time()

        string_list = [f'Average runtime for {name} is {np.mean(value)} seconds \n' for name, value in self.runtimes.items()]
        with open(self.runtime_log_path, 'w') as rtlog:
            [rtlog.write(entry) for entry in string_list]
            rtlog.write('\n')
            rtlog.write(f'Approximate time to run 1 epoch (excluding validation loop): {round(single_epoch_time,2)} hours \n')
            rtlog.write(f'Time to run all {self.opts.epochs} epochs: {round(single_epoch_time*self.opts.epochs, 2)} hours\n')
            rtlog.write(f'Current total runtime: {(current_time - self.init_time)/60} mins')

    

    
