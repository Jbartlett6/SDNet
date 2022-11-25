from torch.utils.tensorboard import SummaryWriter
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
        #self.writer.add_scalar('Deep Regularisation Lambda', net.module.deep_reg, (self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('Validation Fixel Loss', losses['val_fixel_loss']/10,(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        self.writer.add_scalar('Validation Fixel Accuracy', losses['val_fixel_accuracy']/10,(self.dataloader_length*epoch)+i+current_training_details['plot_offset'])
        print(f'[{current_training_details["global_epochs"]+epoch + 1}, {i + 1:5d}] training loss: {losses["running_loss"]/20:.7f} training fod loss {losses["fod_loss"]/20:.7f}')
        

        

class LossTracker():
    def __init__(self,P,criterion):
        self.loss_dict = {'running_loss':0.0,
                         'val_loss':0.0,
                         'acc_loss':0.0,
                         'non_neg':0.0,
                         'fod_loss':0.0, 
                         'fixel_loss':0.0,
                         'fixel_accuracy':0.0,
                         'val_fixel_loss':0.0,
                         'val_fixel_accuracy':0.0}
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
        self.loss_dict['val_fixel_loss'] = 0.0
        self.loss_dict['val_fixel_accuracy'] = 0.0

        
    
    def add_val_losses(self, outputs, labels, val_fixel_loss, val_fixel_accuracy):
        #loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
        loss = self.criterion(outputs.squeeze(), labels)
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
                            'loss_type':opts.loss_type,
                            'learn_lambda':opts.learn_lambda,
                            'Number of Parameters':param_num,
                            'dataset_type':opts.dataset_type}

        with open(os.path.join(model_save_path,'training_details.yml'), 'w') as file:
            documents = yaml.dump(training_details, file)
    
    #Keeping track of the best validation score and implementing early stopping.
        
        

    return current_training_details

def fixel_accuracy(fix_est, gt_fixel):
    fixel_preds = torch.argmax(fix_est, dim = 1)
    val_acc = torch.sum(fixel_preds == gt_fixel)/gt_fixel.shape[0]
    
    return val_acc

    