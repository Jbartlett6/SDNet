import yaml
import os
import argparse
import torch

#with open(os.path.join('config', 'fodnet_config.yml')) as f:
     #config = yaml.load(f, yaml.loader.SafeLoader)
#print(config)

class network_options():
    def __init__(self):
        #Optimisation 
        self.lr = 1e-4
        self.warmup_factor = 1
        self.warmup_epoch_iter = (0,10000)
        self.batch_size = 256
        self.epochs = 1000
        self.val_freq = 20
        
        #Data consistency related hyperparameters
        self.deep_reg = 0.25
        self.learn_lambda = True
        self.fixel_lambda = 0.000160
        #self.fixel_lambda = 0

        #Initialisation Parameters
        self.init_type = None #{'normal', 'xavier', 'kaiming', 'orthogonal'}
        self.activation = 'prelu' #{'relu', 'tanh', 'sigmoid', 'leaky_relu', 'prelu'}

        #Early Stopping Parameters
        self.early_stopping = False
        self.early_stopping_threshold = 0

        #Checkpoint and model parameters
        self.continue_training = False
        self.experiment_name = 'Inference_Baskerville_Test'
        self.model_name = 'best_model.pth'

        #Computation related hyperparameters:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_workers = 16
        self.val_workers = 16
        
        #Data related hyperparameters
        self.data_dir = '/bask/projects/d/duanj-ai-imaging/jxb1336/hcp'
        self.train_subject_list = ['100206']
                    # '100307',
                    # '100408',
                    # '100610',
                    # '101006',
                    # '101107',
                    # '101309',
                    # '101410',
                    # '101915',
                    # '102311',
                    # '102513',
                    # '102614',
                    # '102715',
                    # '102816',
                    # '103010',
                    # '103111',
                    # '103212',
                    # '103414',
                    # '103515',
                    # '103818']
        #102109 has been removed from the test list due to the fixel directory - can be readded providing the fixles are correct.
        self.val_subject_list = ['104012']
                    # '104416',
                    # '104820']

        self.test_subject_list = ['145127', '147737']
                    # '147737',
                    # '174437',
                    # '178849',
                    # '318637',
                    # '581450',
                    # '130821']

        # Dataset parameters.
        self.diffusion_dir = 'Diffusion'
        self.shell_number = 4
        self.data_file = 'normalised_data.nii.gz'
        self.dwi_number = 30
        self.dwi_folder_name = 'undersampled_fod'
        
        self.inference=False
        self.perform_inference=False
        
        self.output_net = False

        #self.option_init()
        
        # if self.continue_training or self.inference:
        #     self.continue_training_init()
        
        print(self.__dict__)
        

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Perform a training loop for the ')

        #Optimisation related hyperparameters.
        parser.add_argument('--lr',type=int, help = 'The initial learning rate used by the ADAM optimiser')
        parser.add_argument('''--warmup_factor',type=int, help = 'The warmup factor used to itinialise the learning rate ()
                            (This is the value the learning rate is multipled by for the first epoch.)''')
        parser.add_argument('--batch_size',type=int, help = 'The batch size which is used to train the model.')
        parser.add_argument('--early_stopping',type=float, help = 'Tur or False option to indicate whether early stopping should take place.')
        parser.add_argument('--early_stopping_threshold',type=float, help = 'The number of ... which have to pass without improvment for early stopping to take place.')
        parser.add_argument('--epochs',type=int, help = 'The number of epochs to train the model for')
        
        #Data consistency related hyperparameters
        parser.add_argument('--deep_reg',type=float, help = 'The deep regularisation parameter which is used in the data consistency term')
        parser.add_argument('--neg_reg',type=float, help = 'The non-negativity regularisation parameter which is used in the data consistency term')
        parser.add_argument('--dc_type',type=str, help = 'The type of data consistency used for the network.')
        parser.add_argument('--learn_lambda',type=bool, help = 'Whether to learn the regularisation parameters in the data consistency layer or not.')
        parser.add_argument('--alpha',type=float, help = 'The smoothing parameter in the network')

        #Network Specific Options:
        parser.add_argument('--model_name', type=str, help = 'The path at which the model is saved')


        #Config/Saving Arguments:
        parser.add_argument('--config_path', type=str, help = 'The path of the config file.')
        parser.add_argument('--continue_training',type=float, help = 'True or False option to indicate whether training should be continued from a previous model or not')
        parser.add_argument('--experiment_name',type=str, help = 'The experiment name, this will be used to create the folder to save the model, as well as the tensorboard logs.')
        

        parser.add_argument('--init_type',type=str, help = 'The type of initialisation used to initialise the weights in the network.')
        parser.add_argument('--activation',type=str, help = 'The type activation function used in the network.')

        self.activation

        #Dataset/subject related hyperparameters:
        parser.add_argument('--data_dir',type=str, help = 'The location of the hcp data, where the subject folders can be found')
        parser.add_argument('--train_subject_list',type=list, help = 'The subject numbers which will be used to train the network')
        parser.add_argument('--val_subject_list',type=list, help = 'The subject numbers which will be used to calculate the validation loss')
        parser.add_argument('--test_subject_list',type=list, help = 'The subject numbers which will be used to obtain test performance metrics at inference time')
        parser.add_argument('--dataset_type',type=str, help = 'Whether to use the full dataset or the experimental dataset')
        parser.add_argument('--subject', type=str, help = 'The path at which the model is saved')


        #Computation/Resource related hyperparameters:
        parser.add_argument('--train_workers',type=int, help = 'The number of workers for the training dataloader')
        parser.add_argument('--val_workers',type=int, help = 'The number of workers for the validation dataloader')
        
        #Inference/Evaluation Parameters:
        parser.add_argument('--perform_inference', type=bool, help = '''When carrying out evaluation this options specifies
                                                                        whether or not to perform inference. i.e. whether the inference 
                                                                        folder already contains the infered data.''')
        parser.add_argument('--inference',type=bool, help = 'Whether the options are being used for inference or not.')

        #HCP specific dataset parameters
        parser.add_argument('--scanner_type',type=str, help = 'Specify whether 3T or 7T data is being used')

        self.parser_args = parser.parse_args()

        for key, value in vars(self.parser_args).items():
            if value != None:
                setattr(self, key, value)

    def config_file(self):
        with open(self.parser_args.config_path, 'r') as f:
            config = yaml.load(f, yaml.loader.SafeLoader)
        
        for key in config.keys():
            setattr(self, key, config[key])
        
    def continue_training_init(self):
        with open(os.path.join('checkpoints', self.experiment_name, 'models', 'training_details.yml')) as f:
            train_details = yaml.load(f, Loader = yaml.loader.SafeLoader)

        for key in train_details.keys():
            if hasattr(self, key):
                setattr(self, key, train_details[key])



    def option_init(self):
        self.parse_arguments()
        if self.parser_args.config_path!=None:
            self.config_file()
            

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
