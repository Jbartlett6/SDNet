import math
import sys

class EarlyStopping():
    '''
    A class to deal with early stopping and learning rate decay in network training. When the 
    early_stopping_update is called, method carries out the following steps:
        1) Checks whether the loss function has improved (compared to self.best_loss)
        2) if the loss function hasn't improved it checks whether it should take action or continue training.
        3) If taking action the learning rate has been decayed too many times (exceeding lr_decay_limit) then the 
        training is stopped. Otherwise the learning rate is decayed.

    '''
    def __init__(self, opts, epoch_length):
        self.early_stopping_counter = 0
        self.best_loss = math.inf
        self.best_loss_iter = 0
        
        
        self.highest_counter = 0
        self.highest_counter_iter = 0
        
        
        self.lr_scheduler_count = 0
        
        self.opts = opts
        self.epoch_length = epoch_length

    def early_stopping_update(self,current_training_details,epoch,i, optimizer):
      
        if self.opts.early_stopping == True:
            current_loss = current_training_details['best_loss']
            current_iter = (self.epoch_length*epoch)+i

            # Checking for improvement.
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.early_stopping_counter = 0
                self.best_loss_iter = current_iter
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.opts.early_stopping_threshold:
                # If the early_stopping_threshold is exceeded then training then either the lr is decayed 
                # or the training is stopped.                 
                
                if self.lr_scheduler_count >= self.opts.lr_decay_limit:
                    # If the limit has been passed more than self.opts.lr_decay_limit times then stop training
                    print(f'Training stopped at epoch {current_training_details["global_epochs"]+epoch} due to Early stopping and minibatch {i}, the best validation loss achieved is: {current_training_details["best_loss"]}')
                    sys.exit()
                else:
                    # Else, decay the learning rate and restart the counter
                    self.early_stopping_counter = 0
                    self.highest_counter = 0
                    self.lr_scheduler_count += 1
                    
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*self.opts.lr_decay_factor
            
            elif self.early_stopping_counter > self.highest_counter:
                self.highest_counter = self.early_stopping_counter
                self.highest_counter_iter = current_iter

        return optimizer 
    
    # def state_dict():
    #             self.early_stopping_counter = 0
    #     self.best_loss = math.inf
    #     self.best_loss_iter = 0
        
        
    #     self.highest_counter = 0
    #     self.highest_counter_iter = 0
        
        
    #     self.lr_scheduler_count = 0
        
    #     self.opts = opts
    #     self.epoch_length = epoch_length
    #     return {}