from cmath import inf
import yaml
import os
import argparse

# with open(os.path.join('config', 'fodnet_config.yml')) as f:
#     config = yaml.load(f, yaml.loader.SafeLoader)
# print(config)

class network_options():
    def __init__(self):
        self.lr = 1e-4
        self.batch_size = 256
        self.epochs = 10

        #Data consistency related hyperparameters
        self.neg_reg = (0.7/0.1875)*0.25
        self.deep_reg = 0.25


        self.early_stopping = True
        self.early_stopping_threshold = inf
        self.continue_training = False
        self.experiment_name = 'CSD_testbed'

        
        #Computation related hyperparameters:
        self.train_workers = 8
        self.val_workers = 8
        self.data_dir = '/media/duanj/F/joe/hcp_2'
        self.train_subject_list = ['100206',
                    '100307',
                    '100408',
                    '100610',
                    '101006',
                    '101107',
                    '101309',
                    '101915',
                    '102109',
                    '102311',
                    '102513',
                    '102614',
                    '102715',
                    '102816',
                    '103010',
                    '103111',
                    '103212',
                    '103414',
                    '103515']
        self.val_subject_list = ['104416',
                        '104012',
                        '103818']

        

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Perform a training loop for the ')

        #Learning related hyperparameters.
        parser.add_argument('--lr',type=int, help = 'The initial learning rate used by the ADAM optimiser')
        parser.add_argument('--batch_size',type=int, help = 'The batch size which is used to train the model.')
        parser.add_argument('--early_stopping',type=float, help = 'Tur or False option to indicate whether early stopping should take place.')
        parser.add_argument('--early_stopping_threshold',type=float, help = 'The number of ... which have to pass without improvment for early stopping to take place.')
        parser.add_argument('--continue_training',type=float, help = 'True or False option to indicate whether training should be continued from a previous model or not')
        parser.add_argument('--experiment_name',type=float, help = 'The experiment name, this will be used to create the folder to save the model, as well as the tensorboard logs.')
        parser.add_argument('--epochs',type=int, help = 'The number of epochs to train the model for')
        #Data consistency related hyperparameters
        parser.add_argument('--deep_reg',type=float, help = 'The deep regularisation parameter which is used in the data consistency term')
        parser.add_argument('--neg_reg',type=float, help = 'The non-negativity regularisation parameter which is used in the data consistency term')

        #Config File path:
        parser.add_argument('--config_path', type=str, help = 'The path of the config file.')

        #Data related hyperparameters:
        parser.add_argument('--data_path',type=str, help = 'The location of the hcp data, where the subject folders can be found')
        parser.add_argument('--train_subject_list',type=str, help = 'The subject numbers which will be used to train the network')
        parser.add_argument('--val_subject_list',type=str, help = 'The subject numbers which will be used to calculate the validation loss')

        #Computation related hyperparameters:
        parser.add_argument('--train_workers',type=int, help = 'The number of workers for the training dataloader')
        parser.add_argument('--val_workers',type=int, help = 'The number of workers for the validation dataloader')

        self.args = parser.parse_args()

    def config_file(self):
        with open(self.args['config_path'], 'r') as f:
            config = yaml.load(f, yaml.loader.SafeLoader)
        
        for key in config.keys():
            setattr(self, key, config[key])
        



    def option_init(self):
        self.config_file()
        self.parse_arguments()    

    def subject_text_reader(self, subject_path):
        with open(subject_path, 'r') as f:
            subs = f.read()
        subs_list = subs.split('\n')
        return subs_list

# parser.add_argument('--model_path', type=str, help = 'The path at which the model is saved')
#     parser.add_argument('--save_path',type=str, help = 'The directory to which to save the final images once inference has been performed.')
#     parser.add_argument('--epochs',type=int, help = 'The number of epochs to train the model for')
#     #parser.add_argument('--lr',type=int, help = 'The initial learning rate used by the ADAM optimiser')
#     #parser.add_argument('--batch_size',type=int, help = 'The batch size which is used to train the model.')
#     #parser.add_argument('--deep_reg',type=float, help = 'The deep regularisation parameter which is used in the data consistency term')
#     #parser.add_argument('--neg_reg',type=float, help = 'The non-negativity regularisation parameter which is used in the data consistency term')
#     #Early stopping needs some work so I can make the units something meaningful, e..g the number of minibatches or epochs rather than 
#     #the number of 20 minibatches which is what it would currently be.
#     #parser.add_argument('--early_stopping',type=float, help = 'The non-negativity regularisation parameter which is used in the data consistency term')
#     parser.add_argument('--lr_decay_rate',type=float, help = 'The exponential learning rate decay parameter - often referred to as gamma')
#     parser.add_argument('--val_freq',type=int, help = 'How often the accuracy of the method is tested on the validation set.')
#     parser.add_argument('--train_workers',type=int, help = 'The number of workers for the training dataloader')
#     parser.add_argument('--val_workers',type=int, help = 'The number of workers for the validation dataloader')

#     # parser.add_argument('--data_path',type=str, help = 'The location of the hcp data, where the subject folders can be found')
#     # parser.add_argument('--train_subject_list',type=str, help = 'The subject numbers which will be used to train the network')
#     # parser.add_argument('--val_subject_list',type=str, help = 'The subject numbers which will be used to calculate the validation loss')
#     # #parser.add_argument('--experiment_name',type=str, help = 'The experiment name, this will be used to create the folder to save the model, as well as the tensorboard logs.')

#     parser.add_argument('--save_frequency',type=int, help = 'After how many evaluations to save the model.')