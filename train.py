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
    def __init__(self, opts):
        
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
        for epoch in range(self.opts.epochs):  # loop over the dataset multiple times

            #The training loop
            for i, data_list in enumerate(self.train_dataloader, 0):
                
                self.lr_warmup_check(epoch, i)
                
                
                inputs, labels, AQ, gt_fixel, _ = data_list
                inputs, labels, AQ, gt_fixel = inputs.to(self.opts.device), labels.to(self.opts.device), AQ.to(self.opts.device), gt_fixel.to(self.opts.device)
                
            
                # zero the parameter gradients and setting network to train
                self.optimizer.zero_grad()
                self.net.train()
                
                #The feeding the data forward through the network.
                
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
                

                
            #Resetting the losses at the end of an epoch to prevent a spike on the graphs.
            self.loss_tracker.reset_losses()
            
            #Updating the early stopping values at the end of the epoch.
            self.current_training_details['previous_loss'] = self.es.early_stopping_update(self.current_training_details,self.opts,epoch,i)
                    
        print('Finished Training')


    def validation_loop(self, epoch, i):
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
        if epoch == 0:
            if i > 10000:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.opts.lr

if __name__ == '__main__':
    opts = options.network_options()
    NT = NetworkTrainer(opts)
    NT.training_loop()

    #Initalising the tensorboard writer
    plt.switch_backend('agg')
    
    #Initialising modules for the network:
    train_dataloader, val_dataloader = data.init_dataloaders(opts)
    criterion = torch.nn.MSELoss(reduction='mean')
    net, P, param_num, current_training_details, model_save_path = Convcsdcfrnet.init_network(opts)
    optimizer = torch.optim.Adam(net.parameters(), lr = opts.warmup_factor*opts.lr, betas = (0.9,0.999), eps = 1e-8)
    loss_tracker = tracker.LossTracker(P,criterion)    
    visualiser = tracker.Vis(opts, train_dataloader)
    es = tracker.EarlyStopping()


    # validation_affine = nib.load(os.path.join(opts.data_dir,'100307','T1w','Diffusion','cropped_fod.nii.gz')).affine
    # print(validation_affine)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    #Initialising the classification network:
    class_network = network.init_fixnet(opts)
    class_criterion = torch.nn.CrossEntropyLoss()
    print(f'The training state of the network is: {class_network.training}')
    print(f'The gradient state of the network is: {class_network.casc[0].weight.requires_grad}')
    
    
    print(optimizer.param_groups[0]['lr'])
    #Running the training loop,in this case for spatial deep reg
    for epoch in range(opts.epochs):  # loop over the dataset multiple times

        #The training loop
        for i, data_list in enumerate(train_dataloader, 0):
            #After one epoch, increase the learning rate
            if epoch == 0:
                if i > 10000:
                    for g in optimizer.param_groups:
                        g['lr'] = opts.lr
            
            inputs, labels, AQ, gt_fixel, _ = data_list
            inputs, labels, AQ, gt_fixel = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device), gt_fixel.to(opts.device)
            
        
            # zero the parameter gradients and setting network to train
            optimizer.zero_grad()
            net.train()
            
            #The feeding the data forward through the network.
            
            outputs = net(inputs, AQ)
            fix_est = class_network(outputs.squeeze()[:,:45])
            

            #Calculating the loss function, backpropagation and stepping the optimizer
            fod_loss = criterion(outputs.squeeze()[:,:45], labels[:,:45])
            fixel_loss = opts.fixel_lambda*class_criterion(fix_est, gt_fixel.long())
            fixel_accuracy = tracker.fixel_accuracy(fix_est, gt_fixel)
            
            #loss = fod_loss+fixel_loss
            loss = fod_loss + fixel_loss
            loss.backward()
            optimizer.step()
            
            # Adding the loss calculated for the current minibatch to the running training loss
            loss_tracker.add_running_loss(loss, fod_loss, fixel_loss, fixel_accuracy)
            

            if i%20 == 19:    
                # #Calculating the average validation loss over 10 random batches from the validation set.
                with torch.no_grad():
                    #Forward pass for calculating validation loss
                    val_temp_dataloader = iter(val_dataloader)
                    for j in range(10):
                        data_list = val_temp_dataloader.next()
                        inputs, labels, AQ, gt_fixel, _ = data_list
                        inputs, labels, AQ, gt_fixel = inputs.to(opts.device), labels.to(opts.device), AQ.to(opts.device), gt_fixel.to(opts.device)

                        #Could put this in a function connected with the model or alternatively put it in a function on its own
                        net.eval()
                        outputs = net(inputs, AQ)

                        #Calculating the fixel based statistics. 
                        fix_est = class_network(outputs.squeeze()[:,:45])
                        fixel_loss = opts.fixel_lambda*class_criterion(fix_est, gt_fixel.long())
                        fixel_accuracy = tracker.fixel_accuracy(fix_est, gt_fixel)

                        

                        loss_tracker.add_val_losses(outputs,labels, fixel_loss, fixel_accuracy)
                        

                #Plotting the results using tensorboard using the visualiser class.
                visualiser.add_scalars(loss_tracker.loss_dict, net, current_training_details, i, epoch)

                #Printing the current best validation loss, and the early stopping counter
                print('Best Loss', current_training_details['best_loss'])
                print('Early stopping counter', es.early_stopping_counter)

                #Updating the training details.
                current_training_details = tracker.update_details(loss_tracker.loss_dict, current_training_details, model_save_path,
                                                            net, epoch, i, opts, optimizer, param_num, train_dataloader)        
                
                #Resetting the losses for the next set of minibatches
                loss_tracker.reset_losses()

                if i%200 == 199:
                    torch.save(net.state_dict(), os.path.join(model_save_path, 'most_recent_model.pth'))
        
        #Resetting the losses at the end of an epoch to prevent a spike on the graphs.
        loss_tracker.reset_losses()
        
        #Updating the early stopping values at the end of the epoch.
        current_training_details['previous_loss'] = es.early_stopping_update(current_training_details,opts,epoch,i)
        
        
                
    print('Finished Training')
        
