'''
Code designed for undersampling diffusion data. 

'''

import torch
import sys
import os
sys.path.append(os.path.join(sys.path[0],'..'))
import nibabel as nib
import sys 

class UndersampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, save_dir, sampling_pattern = [3,9,9,9], bval_samples = [1000, 2000, 3000]):
        
        #Initialising the parameters for the dataset class.
        self.data_path = data_path
        self.save_dir = save_dir

        self.shell_samples_list = sampling_pattern # [#b0 samples, #b1000 samples, #b2000 samples, #b3000 samples]
        self.save_folder='undersampled_fod'

        #Setting the field strength specific parameters
        self.diffusion_dir = 'Diffusion'
        self.shell_number = 4
        self.bval_samples = bval_samples

        #Calculating the mask list and keep lists (using this function here will only work when a constant undersampling pattern is used)
        self.keep_list= self.sample_lists()

        #Creating the data path and the mask path.
        dwi_path = os.path.join(data_path, 'data.nii.gz')
        
        #Loading the data for the subject
        image = nib.load(dwi_path)
        self.head = image.header
        self.aff = image.affine
        print('Loading image data')
        self.image = torch.tensor(image.get_fdata())
        
        #os.mkdir(path = data_path+'/'+subject+'/T1w/Diffusion/undersampled_fod_resptest_1')

    def __len__(self):
        return int(torch.sum(torch.tensor(self.mask)))
    

    def data_save(self):
        '''
        When a constant sampling pattern is used the keep list and mask list can be defined at initalisation as they will be constant for every iteration. However if 
        some aspect of random sampling is used then the keep lists and mask lists will have to be defined in the get item function.
        '''
        print('Saving data')
        im_usamp = nib.Nifti1Image(self.image[:,:,:,self.keep_list].float().detach().numpy(), affine=self.aff)
        save_path = os.path.join(self.save_dir, 'data.nii.gz')

        nib.save(im_usamp, save_path)
        print('Finished saving data')

    def all_save(self):
        self.data_save()
        self.bval_save()
        self.bvec_save()

    def sample_lists(self):
        '''
        A function which returns two lists - the mask_list, which is the q-space volumes to be masked,
        and keep_list for the q-space volumes not to be masked. This will be used in the __getitem__ function 
        of this class.
        '''

        self.bvals = self.bval_extract()

        b0_list = []
        b1000_list = []
        b2000_list = []
        b3000_list = []
    
        for i in range(len(self.bvals)):
            if self.bvals[i] <20:
                b0_list.append(i)
            elif self.bval_samples[0] - 20 < self.bvals[i] < self.bval_samples[0] + 20:
                b1000_list.append(i)
            elif self.bval_samples[1] - 20 < self.bvals[i] < self.bval_samples[1] + 20:
                b2000_list.append(i)
            elif self.bval_samples[2] - 20 < self.bvals[i] < self.bval_samples[2] + 20:
                b3000_list.append(i)


        keep_list = torch.tensor(b0_list[:self.shell_samples_list[0]] +
                                b1000_list[:self.shell_samples_list[1]] +
                                b2000_list[:self.shell_samples_list[2]] +
                                b3000_list[:self.shell_samples_list[3]]
                                )
        return keep_list
    
    def bval_extract(self):
        '''
        Input:
            subj (str) - The subject whose bvals you want to extract.
            path (str) - The path to where the data is located (subjects location).
        
        Desc:
            A function to extract the bvalues from the file they are located in and to return them as a list of values.
        '''
        

        path = os.path.join(self.data_path, 'bvals')
        bvals = open(path, 'r')
        
        bvals_str = bvals.read()
        bvals = [int(b) for b in bvals_str.split()]
        #print('Finished saving bvals')
        return bvals

    def bvec_save(self):
        print('Saving bvectors')
        #Undersampled bvector calculation.
        path = os.path.join(self.data_path, 'bvecs')
        
        with open(path ,'r') as temp:
            bvecs = temp.read()

        bvecsxyz = bvecs.split('\n')
        bvecsxyz.pop(3)

        xvals = [x for x in bvecsxyz[0].split()]
        yvals = [y for y in bvecsxyz[1].split()]
        zvals = [z for z in bvecsxyz[2].split()]

        #Undersampled bvectors
        xvals_new = [xvals[ind] for ind in self.keep_list]
        yvals_new = [yvals[ind] for ind in self.keep_list]
        zvals_new = [zvals[ind] for ind in self.keep_list]

        xvals_str = ' '.join(xvals_new)
        yvals_str = ' '.join(yvals_new)
        zvals_str = ' '.join(zvals_new)

        bvecs_string = '\n'.join((xvals_str,yvals_str,zvals_str))
       
        save_path = os.path.join(self.save_dir, 'bvecs')
        
        with open(save_path, 'w') as temp:
            temp.write(bvecs_string)
        print('Finished Saving bvectors')


    
    def bval_save(self):
        print('Saving bvals')
        ##Bvalues
        new_bvals = [str(self.bvals[ind]) for ind in self.keep_list]
        bvals_string = ' '.join(new_bvals)

        save_path = os.path.join(self.save_dir, 'bvals')

        with open(save_path,'w') as temp:
            temp.write(bvals_string)
        print('Finished saving bvals')


if __name__ == '__main__':
    print('HW')
    diffusion_dir = '/mnt/d/Diffusion_data/CDMD_sub25/sub_025/dwi'
    save_dir = '/mnt/d/Diffusion_data/CDMD_sub25/sub_025/dwi/undersampled_fod'
    UspDset = UndersampleDataset(diffusion_dir, save_dir)
    UspDset.all_save()
    print('HW')