import data
import network
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.join(sys.path[0],'utils'))
import util_funcs

experiment_name = 'sh-bignet'
save_path = os.path.join('checkpoints', experiment_name,'model_dict.pt')
writer_path = os.path.join('checkpoints', experiment_name,'runs')

writer = SummaryWriter(writer_path)
        

#Initailising directories
data_dir = '/media/duanj/F/joe/hcp_2'
train_subject_list = ['100206',
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
'103515',
'103818']

#train_subject_list = ['318637', '100307', '100408']
# train_subject_list = ['318637']
validation_subject_list = ['581450']

#Initialise dataset and dataloaders
train_dataset = data.FODPatchDataset(data_dir, train_subject_list)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2048,
                                            shuffle=True, num_workers=4,
                                            drop_last = True)



validation_dataset = data.FODPatchDataset(data_dir, validation_subject_list)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=256,
                                            shuffle=True, num_workers=4,
                                            drop_last = True)

#Initialising loss, optimiser and model.
criterion = torch.nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
net = network.FixelNet()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, betas = (0.9,0.999), eps = 1e-8)
epochs = 10

val_step = 500
val_loop = 100

#Defining the spherical harmonic basis transfromation:

#sampling_directions = torch.load(os.path.join('utils/300_predefined_directions.pt'))
sampling_directions = torch.load(os.path.join('utils/1281_directions.pt'))
order = 8 
P = util_funcs.construct_sh_basis(sampling_directions, order)
P = P.to(device)
print(P.shape)


#Training Loop
for epoch in range(epochs):
    print(epoch)
    running_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0
    for i, datapoint in enumerate(train_dataloader):
        input_fod, gt_fixel, _ = datapoint
        input_fod, gt_fixel= input_fod.to(device), gt_fixel.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        net.train()

        #feedforward
        #out_fixel = net(torch.matmul(P,input_fod[:,:45].unsqueeze(-1)).squeeze())
        out_fixel = net(input_fod[:,:45])
        
        #loss,backprop and optimizer
        loss = criterion(out_fixel, gt_fixel.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i%val_step == val_step-1:    
            with torch.no_grad():
                for k in range(val_loop):
                    net.eval()
                    val_instance = iter(validation_dataloader)
                    input_fod, gt_fixel, _ = next(val_instance)
                    input_fod, gt_fixel = input_fod.to(device), gt_fixel.to(device)
                    #out_fixel = net(torch.matmul(P,input_fod[:,:45].unsqueeze(-1)).squeeze())
                    out_fixel = net(input_fod[:,:45])
                    loss = criterion(out_fixel, gt_fixel.long())
                    val_loss += loss.item()

                    class_pred = torch.argmax(out_fixel, dim = 1)
                    val_acc += torch.sum(class_pred == gt_fixel)/gt_fixel.shape[0]
                    torch.save(net.state_dict(), save_path)

                    
            
            #Adding the losses to the tensorboard writer.
            step = i + (epoch*len(train_dataloader))
            print(step)
            print(i)
            writer.add_scalar('Training Loss', running_loss/val_step, step)
            writer.add_scalar('Validation Loss', val_loss/val_loop, step)
            writer.add_scalar('Validation Accuracy', val_acc/val_loop, step)

            print(f'The current training loss is: {running_loss/val_step}')
            print(f'The current validation loss is: {val_loss/val_loop}')
            print(f'The current validation accuracy is: {(100*val_acc)/val_loop}%')
            running_loss = 0.0
            val_loss = 0.0
            val_acc = 0.0


#Saving the model state to the save_path. 
torch.save(net.state_dict(), save_path)
    


