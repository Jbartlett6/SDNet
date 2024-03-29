import utils.data as data
import utils.EarlyStopping as EarlyStopping
from models import Convcsdcfrnet
import options
from utils import tracker
from fixel_loss import network

import os 
import sys

import torch 
import torch.optim.lr_scheduler 
import matplotlib.pyplot as plt 
from tqdm import tqdm

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
    def __init__(self, opts: options.NetworkOptions):
        '''
        Description:    Initialising the torch modules which are key for training the network. The dataloaders,
                        SDNet and fixel classification networks are all initialised using this method. 
        '''
        # Loading the network state if specified in network options
        if opts.continue_training == True:
            model_save_path = os.path.join('checkpoints', opts.experiment_name, 'models')
            train_dict = torch.load(os.path.join(model_save_path, 'most_recent_training.pth'))
            self.opts, self.iterations, self.epochs, self.es, net_tuple, self.optimizer = init_from_train_dict(train_dict)
            self.net, self.P, self.param_num, self.current_training_details, self.model_save_path = net_tuple
        else:
            self.opts = opts # cont
            self.es = EarlyStopping.EarlyStopping(self.opts) # cont
            self.net, self.P, self.param_num, self.current_training_details, self.model_save_path = Convcsdcfrnet.init_network(opts) # cont 
            self.optimizer = init_network_optimizer(self.net.parameters(), opts.continue_training, self.opts.warmup_factor*self.opts.lr) # cont
            self.iterations = 0 # cont
            self.epochs = 0
        
        self.train_dataloader, self.val_dataloader = data.init_dataloaders(self.opts)
        self.val_temp_dataloader = iter(self.val_dataloader) 
        
        #Initialising SDNet, criterion and optimiser.
        self.criterion = torch.nn.MSELoss(reduction='mean') 
        
        #Trackers
        self.loss_tracker = tracker.LossTracker(self.criterion)    
        self.visualiser = tracker.Vis(self.opts, self.train_dataloader)
        self.init_runtime_trackers(runtime_mem = 5)

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
        for epoch in range(self.epochs, self.opts.epochs):  # loop over the dataset multiple times

            #The training loop
            for i, data_list in enumerate(tqdm(self.train_dataloader)):
                
                #Stopping training if the number of iterations exceeds the iteration limit.
                if self.iterations>=opts.iteration_limit:
                    print('Stopped training due to reaching the iteration limit')
                    sys.exit()

                self.rttracker.stop_timer('training dataload')
                # Checking whether leraning rate warm up has ended
                self.rttracker.start_timer('training iter')
                self.lr_warmup_check(self.iterations)
                
                #Loading data to GPUs
                inputs, labels, AQ, gt_fixel, _ = data_list
                inputs, labels, AQ, gt_fixel = inputs.to(self.opts.device), labels.to(self.opts.device), AQ.to(self.opts.device), gt_fixel.to(self.opts.device)
            
                # Zero the parameter gradients and setting network to train
                self.optimizer.zero_grad()
                self.net.train()
                
                self.rttracker.start_timer('sdnet forward pass')
                outputs = self.net(inputs, AQ)
                self.rttracker.stop_timer('sdnet forward pass')
                self.rttracker.start_timer('fix forward pass')
                fix_est = self.class_network(outputs.squeeze()[:,:45])
                self.rttracker.stop_timer('fix forward pass')
                #Calculating the loss function, backpropagation and stepping the optimizer
                fod_loss = self.criterion(outputs.squeeze()[:,:45], labels[:,:45])
                fixel_loss = self.opts.fixel_lambda*self.class_criterion(fix_est, gt_fixel.long())
                fixel_accuracy = tracker.fixel_accuracy(fix_est, gt_fixel)
                
                #loss = fod_loss+fixel_loss
                loss = fod_loss + fixel_loss

                self.rttracker.start_timer('grad and step')
                loss.backward()
                self.optimizer.step()
                self.rttracker.stop_timer('grad and step')

                # Adding the loss calculated for the current minibatch to the running training loss
                self.loss_tracker.add_running_loss(loss, fod_loss, fixel_loss, fixel_accuracy)
                self.rttracker.stop_timer('training iter')

                if i%self.opts.val_freq == self.opts.val_freq-1:
                    self.rttracker.start_timer('validation loop')
                    self.validation_loop(epoch, i)
                    self.rttracker.stop_timer('validation loop')
                    self.optimizer = self.es.early_stopping_update(self.current_training_details, self.iterations, self.optimizer)
                
                self.rttracker.write_runtimes()
                self.rttracker.start_timer('training dataload')
                self.iterations += 1
                
            self.loss_tracker.reset_losses()
            
                    
        print('Finished Training')


    def validation_loop(self, epoch, i):
        '''
        Description:    The validation loop that is called within the training loop. This method validates
                        the performance of the network on the validation subjects as specified in the
                        options.py script. 
        '''
        with torch.no_grad():
            
            for j in range(self.opts.val_iters):
                
                self.rttracker.start_timer('validation iter')
                #Loading this iterations data into the GPU
                
                data_list = next(self.val_temp_dataloader, 'reset_val_dataloader')
                if data_list == 'reset_val_dataloader':
                    self.val_temp_dataloader = iter(self.val_dataloader)
                    data_list = next(self.val_temp_dataloader)

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
                self.rttracker.stop_timer('validation iter')

        self.rttracker.start_timer('post val steps')        
        #Plotting the results using tensorboard using the visualiser class.
        self.visualiser.add_scalars(self.loss_tracker.train_loss_dict, self.loss_tracker.val_loss_dict, self.current_training_details, epoch, self.iterations)

        #Printing the current best validation loss, and the early stopping counter
        print('Best Loss', self.current_training_details['best_loss'])
        print('Early stopping counter', self.es.early_stopping_counter)

        #Updating the training details.
        self.current_training_details = tracker.update_training_logs(self.loss_tracker.train_loss_dict, self.loss_tracker.val_loss_dict, self.current_training_details, self.model_save_path,
                                                    self.net, epoch, i, self.opts, self.optimizer, self.param_num, self.train_dataloader, self.es, self.iterations)        
        
        #Resetting the losses for the next set of minibatches
        self.loss_tracker.reset_losses()

        
        training_state_dict = {'net_state': self.net.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'earlystopping_state': self.es.state_dict(),
            'epochs': epoch,
            'iterations':self.iterations,
            'opts': self.opts}
        
        torch.save(training_state_dict, os.path.join(self.model_save_path, 'most_recent_training.pth'))

        self.rttracker.stop_timer('post val steps')


    def lr_warmup_check(self, iteration):
        '''
        Description:    Checking whether the learning rate warm up period has ended. Initially the learning rate 
                        will be set to the warmup factor*learning rate. Once the network has been training for 
                        opts.warmup_iter iterations the learning rate will increase to its final value. 
        '''

        if iteration> opts.warmup_iter:
            for g in self.optimizer.param_groups:
                g['lr'] = self.opts.lr

    def init_runtime_trackers(self, runtime_mem):
        self.rttracker = tracker.RuntimeTracker(runtime_mem, os.path.join('checkpoints', opts.experiment_name , 'logs', 'runtime.log'), self.opts, len(self.train_dataloader))
        self.rttracker.add_runtime_tracker('training iter')
        self.rttracker.add_runtime_tracker('validation iter')
        self.rttracker.add_runtime_tracker('validation loop')

        self.rttracker.add_runtime_tracker('post val steps')

        self.rttracker.add_runtime_tracker('sdnet forward pass')
        self.rttracker.add_runtime_tracker('fix forward pass')
        self.rttracker.add_runtime_tracker('grad and step')
        self.rttracker.add_runtime_tracker('training dataload')

def init_network_optimizer(params, continue_training_bool, initial_lr):
    '''
    Function to load the initial optimiser. If training is continuing from a pre-trained value, 
    the optimiser's state is loaded fom that which was saved when training was paused. 
    '''
    optimizer = torch.optim.Adam(params, lr = initial_lr, betas = (0.9,0.999), eps = 1e-8)

    # If continuing training load the best optimiser from the model_dict.
    if continue_training_bool:
        model_save_path = os.path.join('checkpoints', opts.experiment_name, 'models')
        optimizer.load_state_dict(torch.load(os.path.join(model_save_path,'best_training.pth'))['optim_state'])
    
    return optimizer

def init_from_train_dict(train_dict):
    """Initialise attributes of the training loop from train_dict 

    Given a train_dict, the attributes which are dependent on the 
    contents of the dictionary are loaded. Those attributes are:
    * self.opts
    * self.iterations
    * self.es
    * net_tuple (net and other related attributes)
     

    Args:
        train_dict (dict): the training state dict as saved from a 
        previous training run
    """    
    opts = train_dict['opts']
    opts.continue_training=True
    iterations = train_dict['iterations']
    epochs = train_dict['epochs']

    es = EarlyStopping.EarlyStopping(opts)
    es.load_state_dict(train_dict['earlystopping_state'])  

    net_tuple = Convcsdcfrnet.init_network(opts) # cont 

    optimizer = init_network_optimizer(net_tuple[0].parameters(), True, opts.lr) # cont

    return opts, iterations, epochs, es, net_tuple, optimizer
        

if __name__ == '__main__':
    plt.switch_backend('agg')
    opts = options.NetworkOptions()
    NT = NetworkTrainer(opts)
    NT.training_loop()