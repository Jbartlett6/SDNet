import sys
import os 
sys.path.append(os.path.join(sys.path[0],'models'))
sys.path.append(os.path.join(sys.path[0],'utils'))
import util
import torch 
import matplotlib.pyplot as plt 
import data
import options
#import FODCSDNet as csdnet
import Convcsdnet
import Convcsdcfrnet
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler 
import tracker 
import nibabel as nib
sys.path.append(os.path.join(sys.path[0],'..', 'fixel_loss'))
import network

class NetworkTrainer():
    '''
    Description:    Class to carry out network training accoridng to the configuration options 
                    defined in options.py as well as the various pytorch modules required within the class such as
                    data, network etc. 
    Methods:
                __init__()      - Initialising the torch modules which are key for training the network. The dataloaders,
                                SDNet and fixel classification networks are all initialised using this method. 
                training_loop   - The training loop   
                validation_loop - The validation loop that is called within the training loop. This method validates
                                the performance of the network on the validation subjects as specified in the
                                options.py script.
    '''
    def __init__(self, opts):
        '''
        Description:    Initialising the torch modules which are key for training the network. The dataloaders,
                        SDNet and fixel classification networks are all initialised using this method. 
        '''
        
        #Initialising modules for the network:
        self.opts = opts
        self.train_dataloader, self.val_dataloader = data.init_dataloaders(self.opts)
        
        #Initialising SDNet, criterion and optimiser.
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.net, self.P, self.param_num, self.current_training_details, self.model_save_path = Convcsdcfrnet.init_network(opts)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.opts.warmup_factor*self.opts.lr, betas = (0.9,0.999), eps = 1e-8)
        
        #Initialising trackers
        self.loss_tracker = tracker.LossTracker(self.P,self.criterion)    
        self.visualiser = tracker.Vis(self.opts, self.train_dataloader)
        self.es = tracker.EarlyStopping()

        #Initialising the classification network:
        self.class_network = network.init_fixnet(self.opts)
        self.class_criterion = torch.nn.CrossEntropyLoss()
        print(f'The training state of the network is: {self.class_network.training}')
        print(f'The gradient state of the network is: {self.class_network.casc[0].weight.requires_grad}')
        print(self.optimizer.param_groups[0]['lr'])

    def training_loop(self):
        '''
        Description:    The training loop
        '''
        for epoch in range(self.opts.epochs):  # loop over the dataset multiple times

            #The training loop
            for i, data_list in enumerate(self.train_dataloader, 0):
                
                # Checking whether leraning rate warm up has ended
                self.lr_warmup_check(epoch, i)
                
                #Loading data to GPUs
                inputs, labels, AQ, gt_fixel, _ = data_list
                inputs, labels, AQ, gt_fixel = inputs.to(self.opts.device), labels.to(self.opts.device), AQ.to(self.opts.device), gt_fixel.to(self.opts.device)
            
                # Zero the parameter gradients and setting network to train
                self.optimizer.zero_grad()
                self.net.train()
                outputs = self.net(inputs, AQ)
                fix_est = self.class_network(outputs.squeeze()[:,:45])
                
                #Calculating the loss function, backpropagation and stepping the optimizer
                fod_loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
                fixel_loss = self.opts.fixel_lambda*self.class_criterion(fix_est, gt_fixel.long())
                fixel_accuracy = tracker.fixel_accuracy(fix_est, gt_fixel)
                
                #loss = fod_loss+fixel_loss
                loss = fod_loss + fixel_loss
                loss.backward()
                self.optimizer.step()
                
                # Adding the loss calculated for the current minibatch to the running training loss
                self.loss_tracker.add_running_loss(loss, fod_loss, fixel_loss, fixel_accuracy)

                if i%self.opts.val_freq == self.opts.val_freq-1:
                    print(f'Epoch:{epoch}, Minibatch:{i}/{len(self.train_dataloader)}')
                    self.validation_loop(epoch, i)
                    print('validation')
                
            self.loss_tracker.reset_losses()
            self.current_training_details['previous_loss'] = self.es.early_stopping_update(self.current_training_details,self.opts,epoch,i)
                    
        print('Finished Training')


    def validation_loop(self, epoch, i):
        '''
        Description:    The validation loop that is called within the training loop. This method validates
                        the performance of the network on the validation subjects as specified in the
                        options.py script. 
        '''
        with torch.no_grad():

            #Forward pass for calculating validation loss
            val_temp_dataloader = iter(self.val_dataloader)
            for j in range(10):

                #Loading this iterations data into the GPU
                data_list = val_temp_dataloader.next()
                inputs, labels, AQ, gt_fixel, _ = data_list
                inputs, labels, AQ, gt_fixel = inputs.to(self.opts.device), labels.to(self.opts.device), AQ.to(self.opts.device), gt_fixel.to(self.opts.device)

                #Could put this in a function connected with the model or alternatively put it in a function on its own
                self.net.eval()
                outputs = self.net(inputs, AQ)

                #Calculating the fixel based statistics. 
                fix_est = self.class_network(outputs.squeeze()[:,:45])
                fixel_loss = self.opts.fixel_lambda*self.class_criterion(fix_est, gt_fixel.long())
                fixel_accuracy = tracker.fixel_accuracy(fix_est, gt_fixel)

                self.loss_tracker.add_val_losses(outputs,labels, fixel_loss, fixel_accuracy)
                
        #Plotting the results using tensorboard using the visualiser class.
        self.visualiser.add_scalars(self.loss_tracker.loss_dict, self.net, self.current_training_details, i, epoch)

        #Printing the current best validation loss, and the early stopping counter
        print('Best Loss', self.current_training_details['best_loss'])
        print('Early stopping counter', self.es.early_stopping_counter)

        #Updating the training details.
        self.current_training_details = tracker.update_details(self.loss_tracker.loss_dict, self.current_training_details, self.model_save_path,
                                                    self.net, epoch, i, self.opts, self.optimizer, self.param_num, self.train_dataloader)        
        
        #Resetting the losses for the next set of minibatches
        self.loss_tracker.reset_losses()

        if i%200 == 199:
            torch.save(self.net.state_dict(), os.path.join(self.model_save_path, 'most_recent_model.pth'))


    def lr_warmup_check(self, epoch, i):
        '''
        Description:    Checking whether the learning rate warm up period has ended. Initially the llearning rate 
                        will be set to the warmup factor*learning rate. Once the network has been training for 
                        opts.warmup_epoch_iter[0] epochs and opts.warmup_epoch_iter[1] iterations the learning 
                        rate will increase to its final value. 
        '''
        if epoch == opts.warmup_epoch_iter[0]:
            if i > opts.warmup_epoch_iter[1]:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.opts.lr

if __name__ == '__main__':
    plt.switch_backend('agg')
    opts = options.network_options()
    NT = NetworkTrainer(opts)
    NT.training_loop()