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
    def __init__(self, opts):
        self.early_stopping_counter = 0
        self.best_loss = math.inf
        self.best_loss_iter = 0
        
        
        self.highest_counter = 0
        self.highest_counter_iter = 0
        
        
        self.lr_scheduler_count = 0
        
        self.opts = opts

    def early_stopping_update(self,current_best_loss, epoch, current_iter, optimizer):
      
        if self.opts.early_stopping == True:
            

            # Checking for improvement.
            if current_best_loss < self.best_loss:
                self.best_loss = current_best_loss
                self.early_stopping_counter = 0
                self.best_loss_iter = current_iter
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.opts.early_stopping_threshold:
                # If the early_stopping_threshold is exceeded then training then either the lr is decayed 
                # or the training is stopped.                 
                
                if self.lr_scheduler_count >= self.opts.lr_decay_limit:
                    # If the limit has been passed more than self.opts.lr_decay_limit times then stop training
                    print(f'Training stopped at epoch {epoch}, total iteration {current_iter} due to , the best validation loss achieved is: {current_best_loss}')
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
    
    def state_dict(self):
        """Returns the current state dict of the class.

        The state dict of the EarlyStopping class contains:
        * early_stopping_counter - how long since the loss has improved 
        * best_loss - The best loss achieved by the model
        * best_loss_iter - The iteration where the best loss occured
        * highest_counter - The highest the early stopping counter has been
        since being reset
        * highest_counter_iter - The iteration thhat highesst_conter was 
        recorded at 
        * lr_scheduler_count - How many times the lr scheduler has updated
        the lr

        Returns:
            dict: The state dict
        """
        state_dict = {'early_stopping_counter': self.early_stopping_counter,
                      'best_loss': self.best_loss,
                      'best_loss_iter': self.best_loss_iter,
                      'highest_counter': self.highest_counter,
                      'highest_counter_iter': self.highest_counter_iter,
                      'lr_scheduler_count': self.lr_scheduler_count}
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Loads a state dict

        Args:
            state_dict (dict): The state_dict to load 
        """        
        for key, val in state_dict.items():
            setattr(self, key, val)