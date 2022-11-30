import sys
import os 
sys.path.append(os.path.join(sys.path[0],'..'))
import torch 
import torch.nn as nn
import util 
import netblocks
import math
import yaml


class CSDNet(nn.Module):
    def __init__(self, opts):
        super(CSDNet, self).__init__()

        self.opts = opts
        #self.I = torch.eye(47).to(opts.device)
        self.register_buffer('I', torch.eye(47))
        #Initialising the data consistency parameters.
        self.init_dc_params()

        #Not currently used (needs tidying).
        self.sampling_directions = torch.load(os.path.join('utils/300_predefined_directions.pt'))
        self.order = 8 
        P = util.construct_sh_basis(self.sampling_directions, self.order)
        P_temp = torch.zeros((300,2))
        self.register_buffer('P',torch.cat((P,P_temp),1))
          
        activation_mod = self.set_activation()

        self.csdcascade_1 = netblocks.SHConvCascadeLayer(activation_mod)
        self.csdcascade_2 = netblocks.SHConvCascadeLayer(activation_mod)
        self.csdcascade_3 = netblocks.SHConvCascadeLayer(activation_mod)
        self.csdcascade_4 = netblocks.SHConvCascadeLayer(activation_mod)
        #self.output_net = netblocks.OutputLayer()

        if self.opts.output_net:
            self.output_net = netblocks.OutputLayer()


        
        # self.csdcascade_1 = netblocks.GLUConvCascadeLayer()
        # self.csdcascade_2 = netblocks.GLUConvCascadeLayer()
        # self.csdcascade_3 = netblocks.GLUConvCascadeLayer()
        # self.csdcascade_4 = netblocks.GLUConvCascadeLayer()
        
        self.init_weight(self.opts.activation)
        
        
    def forward(self, b, AQ):
        #Initialising some matricies which will be used throughout the forward pass for given data.
        AQ_Tb = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),b)
        AQ_TAQ = torch.matmul(AQ.transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1),AQ.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        
        #Initialising c using only lower order spsheircal harmonics
        c = self.c_init(AQ, AQ_Tb, b)
        #dc = c
        
        
        #First cascade
        c_csd = self.csdcascade_1(c)
        # curr_feat = torch.zeros([128, 512, 9,9,9]).to(b.device)
        # c_csd, curr_feat = self.csdcascade_1(c_inp, curr_feat)
        
        c_csd = torch.mul(c_csd[:,:,:,:,:47,:], torch.sigmoid(c_csd[:,:,:,:,47:,:]))
        c = self.dc(c, c_csd, AQ_Tb, AQ_TAQ, b,1)
        # c_cat = torch.cat((c,dc[:,1:-1,1:-1,1:-1,:]), dim = 4)
        # dc = c
        
        #Second Cascade
        c_csd = self.csdcascade_2(c)
        # c_csd, curr_feat = self.csdcascade_2(c_inp, curr_feat)
        c_csd = self.res_con(c_csd,c)
        c = self.dc(c, c_csd, AQ_Tb, AQ_TAQ, b,2)
        # c_cat = torch.cat((c,dc[:,1:-1,1:-1,1:-1,:]), dim = 4)
        # dc = c
        
        
        #Third Cascade
        c_csd = self.csdcascade_3(c)
        # c_csd, curr_feat = self.csdcascade_3(c_inp, curr_feat)
        c_csd = self.res_con(c_csd,c)
        c = self.dc(c, c_csd, AQ_Tb, AQ_TAQ, b,3)
        # c_cat = torch.cat((c,dc[:,1:-1,1:-1,1:-1,:]), dim = 4)
        # dc = c
        
        
        #Final Cascade
        c_csd = self.csdcascade_4(c)
        # c_csd, curr_feat = self.csdcascade_4(c_inp, curr_feat)
        c_csd = self.res_con(c_csd,c)
        c = self.dc(c,c_csd, AQ_Tb, AQ_TAQ, b,4)
       
        # #A small network to just allow the network extra freedom
        # c = c.squeeze()
        # c = c + self.output_net(c)
        # c = c.unsqueeze(-1)

        return c

    def dc(self, c, c_csd, AQ_Tb, AQ_TAQ, b,n):
        '''
        The data consistency block used in the network.
        '''
        c = c[:,1:-1,1:-1,1:-1,:,:]
        A_tmp = (AQ_TAQ+self.deep_reg*self.I)
        b_tmp = (AQ_Tb[:,n:-n,n:-n,n:-n,:,:]+self.deep_reg*c_csd)
        c = torch.linalg.solve(A_tmp, b_tmp)
        return c

    def res_con(self,c_csd, c):
        '''
        The residual connections. This layer serves to add the residual,
        (the output of the cascade) to the output of the previous data consistency term.
        '''
        c_csd = c[:,1:-1,1:-1,1:-1,:,:] + torch.mul(c_csd[:,:,:,:,:47,:], torch.sigmoid(c_csd[:,:,:,:,47:,:]))
        return c_csd
    
    def c_init(self, AQ, AQ_Tb, b):
        '''
        Initialises c before the first cascade. This method uses the lower order spherical harmonics 
        as is done by the MSMT CSD algorithm in MRtrix 3 software package. An alternative would be to 
        use the full spherical harmonics first.
        '''
        ##AQ = [B,30,47] ---> AQ_TAQ_temp = [B, 18,18]
        AQ_TAQ_temp = torch.matmul(AQ[:,:,[i for i in range(16)]+[45,46]].transpose(1,2).unsqueeze(1).unsqueeze(1).unsqueeze(1), AQ[:,:,[i for i in range(16)]+[45,46]].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        c_hat = torch.linalg.solve(AQ_TAQ_temp+0.01*torch.eye(18).to(b.device),AQ_Tb[:,:,:,:,[i for i in range(16)]+[45,46],:])
        
        #c_hat = [B,X,Y,Z,18,1] ---> c = [B,X,Y,Z,47,1]
        c = torch.zeros((c_hat.shape[0], c_hat.shape[1], c_hat.shape[2], c_hat.shape[3],47,1)).to(b.device)
        c[:,:,:,:,:16,:] = c_hat[:,:,:,:,:16,:]
        c[:,:,:,:,45:,:] = c_hat[:,:,:,:,16:,:]
    
        return c

    def init_weight(self,activation ,init_gain = 1.0):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                if self.opts.init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif self.opts.init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
                elif self.opts.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity=activation)
                elif self.opts.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=init_gain)
    
    def set_activation(self):
        if self.opts.activation == 'sigmoid':
            mod = nn.Sigmoid()
        elif self.opts.activation == 'relu':
            mod = nn.ReLU(inplace=True)
        elif self.opts.activation == 'tanh':
            mod = nn.Tanh()
        elif self.opts.activation == 'leaky_relu':
            mod = nn.LeakyReLU(inplace = True)
        elif self.opts.activation == 'prelu':
            mod = torch.nn.PReLU(num_parameters=1, init=0.25, device=self.opts.device)
        return mod

    def init_dc_params(self):
        if self.opts.learn_lambda == True:
            self.deep_reg = nn.Parameter(torch.tensor(self.opts.deep_reg))
            self.neg_reg = nn.Parameter(torch.tensor(self.opts.neg_reg))
            self.alpha = nn.Parameter(torch.tensor(self.opts.alpha).float())
        else:
            #Data consistency term parameters:
            self.register_buffer('deep_reg', torch.tensor(self.opts.deep_reg))
            self.register_buffer('neg_reg', torch.tensor(self.opts.neg_reg))
            self.register_buffer('alpha', torch.tensor(self.opts.alpha).float())

        
                


def init_network(opts):
    #Initialising the network and moving it to the correct device.
    print('Initialising Network')
    net = CSDNet(opts)
    P = net.P.to(opts.device)
    net = nn.DataParallel(net)
    net = net.to(opts.device)
    
    #Loading state dict (Identity has to be added due to changes made to the network) 
    loaded_state_dict = torch.load('/bask/projects/d/duanj-ai-imaging/jxb1336/code/CSDNet/checkpoints/deep_sh_casc/models/best_model.pth')
    loaded_state_dict['module.I'] = torch.eye(47)
    net.load_state_dict(loaded_state_dict)
    
    #Printing the layers and number of parameters of the network.
    print(net)
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'The number of parameters in the model is: {param_num}')

    model_save_path = os.path.join('checkpoints', opts.experiment_name, 'models')
    current_training_details = {'plot_offset':0, 'previous_loss':math.inf, 'best_loss':math.inf, 'best_val_ACC':0, 'global_epochs':0}
    if opts.continue_training:
        assert os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), 'The experiment ' + opts.experiment_name + ''' does not exist so model parameters cannot be loaded. 
                                                                            Either change continue training flag to create another experiment, or change the experiment name
                                                                            to load an existing experiment'''

        net.load_state_dict(torch.load(os.path.join(model_save_path,'best_model.pth')))
        
        with open(os.path.join(model_save_path,'training_details.yml'), 'r') as file:
            training_details = yaml.load(file, yaml.loader.SafeLoader)
        current_training_details['plot_offset'] = training_details['plot_step']
        current_training_details['best_loss'] = training_details['best loss']
        current_training_details['previous_loss'] = training_details['best loss']
        current_training_details['best_val_ACC'] = training_details['best ACC']
        current_training_details['global_epochs'] = training_details['epochs_count']
        current_training_details['plot_offset'] = training_details['plot_step']

        print('Plot offset is:'+str(current_training_details['plot_offset']))
        
        
    else:
        assert not os.path.isdir(os.path.join('checkpoints', opts.experiment_name)), f'The experiment {opts.experiment_name} already exists, please select another experiment name'
        os.mkdir(os.path.join('checkpoints', opts.experiment_name))
        os.mkdir(os.path.join('checkpoints', opts.experiment_name, 'models'))

    return net, P, param_num, current_training_details, model_save_path
  






