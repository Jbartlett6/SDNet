import torch
import math

class NetworkOptions():
    def __init__(self):
        ### General ###
        self.experiment_name = 'tong_SDNet_1_25'  # Directory name to be stored in the checkpoint directory. **General**
        self.model_name = 'best_model.pth'  # Working model name - will be stored within the experiment name directory found within the ** General **
                                            # checkpoints directory. 
        self.continue_training = True      # Whether to continue training using existing model weights

        self.inference=True
        self.perform_inference=True

        ### Training Options ###
        self.lr = 1e-4                      # Learning rate - set to this value post learning rate warmup. 
        self.warmup_factor = 1e-2              # When the network is warming up the effective learning rate is set to warmup_factor*lr
        self.warmup_iter = 10000             # Number of iterations after which the network stops warming up.
        self.batch_size = 256               # Training batch_size.
        
        self.val_freq = 500                 # How often (iterations) to run the validation loop inside the training loop
        self.val_iters = 10                 # How many iterations of validation data to loop through when the validation loop is called.
        
        # Stopping Conditions
        self.epochs = 100                   # Maxium number of epochs
        self.iteration_limit = 202505     # Maximum number of iterations (training updates)
        self.lr_decay_limit = math.inf      # The number of times to decay the learning rate 
        self.lr_decay_factor = 1            # The learning rate decay factor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # **General**
        self.train_workers = 8            # Number of workers used for the training dataloader.
        self.val_workers = 8 # Number of workers used for the validation dataloader. 

        #Early Stopping Parameters
        self.early_stopping = False          # Whether to include early stopping in the network.
        self.early_stopping_threshold = math.inf  # When early_stopping_counter reaches this value training will stop. This counter is updated every validation loop, therefore
                                            # training will be stopped due to early stopping when the network hasn't improved for early_stopping_threshold validation loops. 

        ### Network ###
        self.deep_reg = 0.25                # The deep regularisation parameter. If learn_lambda = True this is only the initial value. 
        self.learn_lambda = True            # Whether to optimise lambda within the network.
        self.fixel_lambda = 0               # kappa.

        self.init_type = None               # {'normal', 'xavier', 'kaiming', 'orthogonal'}
        self.activation = 'prelu'           # {'relu', 'tanh', 'sigmoid', 'leaky_relu', 'prelu'}

        ### Data ###
        # self.data_dir = '/media/duanj/F/joe/hcp_2' # Data directory see github repo for more details.
        self.data_dir = '/bask/projects/d/duanj-ai-imaging/jxb1336/Tong_as_HCP_1_25'

        self.train_subject_list = ['113821',
                    '523032',
                    '130518',
                    '151021',
                    '130417',
                    '130720',
                    '202113',
                    '188751',
                    '118225',
                    '284646',
                    '120111',
                    '123824',
                    '268850',
                    '123117',
                    '368753',
                    '161327',
                    '176845',
                    '159441',
                    '559457',
                    '115017']
        
        #102109 has been removed from the test list due to the fixel directory - can be readded providing the fixles are correct.
        self.val_subject_list = ['104012',  # Subjects to be used for validation
                    '104416',
                    '104820']

        self.diffusion_dir = 'Diffusion'
        self.shell_number = 4
        self.data_file = 'normalised_data.nii.gz'
        self.dwi_number = 30
        self.dwi_folder_name = 'undersampled_fod'

        ### Inference ###
        # self.test_subject_list = ['145127', # Subjects to be used for testing. **Inference**
        #     '147737',
        #     '174437',
        #     '178849',
        #     '318637',
        #     '581450',
        #     '130821']
        self.test_subject_list = ['1', '2', '3']

        print(self.__dict__)
