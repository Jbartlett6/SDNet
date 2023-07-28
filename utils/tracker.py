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
                add_scalars     
    '''
    def __init__(self, opts, train_dataloader):
        self.writer = SummaryWriter(os.path.join('checkpoints', opts.experiment_name,'runs'))
        self.dataloader_length = len(train_dataloader)
        
    def add_scalars(self,losses,net,current_training_details,i,epoch):
        step = (self.dataloader_length*epoch)+i+current_training_details['plot_offset']
        # step = (self.dataloader_length*epoch)+i
        #Training loss and its decomposition :
        self.writer.add_scalar('Training Loss', losses['running_loss']/20, step)
        self.writer.add_scalar('FOD Loss', losses['fod_loss']/20, step)
        self.writer.add_scalar('Fixel Loss', losses['fixel_loss']/20, step)
        self.writer.add_scalar('Fixel Accuracy', losses['fixel_accuracy']/20, step)
        
        #The validation losses
        self.writer.add_scalar('Validation Loss', losses['val_loss']/10,step)
        self.writer.add_scalar('Validation ACC', losses['acc_loss']/10 ,step)        
        #self.writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, step)
        self.writer.add_scalar('Validation Fixel Loss', losses['val_fixel_loss']/10,step)
        self.writer.add_scalar('Validation Fixel Accuracy', losses['val_fixel_accuracy']/10,step)
        print(f'[{current_training_details["global_epochs"]+epoch + 1}, {i + 1:5d}] training loss: {losses["running_loss"]/20:.7f} training fod loss {losses["fod_loss"]/20:.7f}')
        

    
    
        

        

class LossTracker():
    def __init__(self,P,criterion):
        loss_keys = ['running_loss',
                         'val_loss',
                         'acc_loss',
                         'non_neg',
                         'fod_loss', 
                         'fixel_loss',
                         'fixel_accuracy',
                         'val_fixel_loss',
                         'val_fixel_accuracy']

        self.loss_dict = dict.fromkeys(loss_keys, 0.0)
        self.P = P
        self.criterion=criterion

    def reset_losses(self):
        self.loss_dict = dict.fromkeys(self.loss_dict.keys(), 0.0)

    
    def add_val_losses(self, outputs, labels, val_fixel_loss, val_fixel_accuracy):
        loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
        #loss = self.criterion(outputs.squeeze(), labels)
        self.loss_dict['val_loss'] += loss.item()
        self.loss_dict['acc_loss'] += util.ACC(outputs,labels).mean()
        self.loss_dict['non_neg'] += torch.sum((torch.matmul(self.P, outputs)<-0.01).squeeze(),axis = -1).float().mean()
        self.loss_dict['val_fixel_loss'] += val_fixel_loss.item()
        self.loss_dict['val_fixel_accuracy'] += val_fixel_accuracy.item()

    def add_running_loss(self,loss,fod_loss,fixel_loss, fixel_accuracy):
        self.loss_dict['running_loss'] += loss.item()
        self.loss_dict['fod_loss'] += fod_loss.item()
        self.loss_dict['fixel_loss'] += fixel_loss.item()
        self.loss_dict['fixel_accuracy'] += fixel_accuracy.item()



def update_details(losses, current_training_details, model_save_path, net, epoch, i, opts, optimizer, param_num, train_dataloader):
    if losses['acc_loss']/10 > current_training_details['best_val_ACC']:
        current_training_details['best_val_ACC'] = losses['acc_loss']/10

    if losses['val_loss']/10 < current_training_details['best_loss']:
                    
        current_training_details['best_loss'] = losses['val_loss']/10
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

        with open(os.path.join(model_save_path,'training_details.yml'), 'w') as file:
            documents = yaml.dump(training_details, file)
    
    #Keeping track of the best validation score and implementing early stopping.
        
        

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

    

    
