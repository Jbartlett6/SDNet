# CSDNet
![alt text]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg](https://github.com/Jbartlett6/CSDNet/blob/master/SDNet%20(2).jpg)?raw=true)



The directory the code comes in is named CSDNet_dir, and will have the following structure:
```bash
.
├── data.py
├── util.py
└── inference checkpoints
    └── experiment_name
        └── inference
        └── runs
        └── model_saves
└── models
    └── csdnet
└── model_app
    ├── inference.py
    └── train.py
```
The structure of the data directory is as follows for training subject. The data directory contains all of the subjects which you have access to. Which of these subjects are used for training, validation and testing is then determined by lists which are input into the script/dataset. The subjects in the test set are much the same only with an additional directory 'tractseg' in the Diffusion folder which is used for evaluation in certain white matter tracts. 

```bash
.
└── data_directory
    └── subject
        └── T1w
            └── T1w_acpc_dc_restore_1.25.nii.gz
            └── 5ttgen.nii.gz
            └── white_matter_mask.nii.gz
            └── Diffusion
                └── bvals
                └── bvecs 
                └── data.nii.gz
                └── nodif_brain_mask.nii.gz
                └── normalised_data.nii.gz
                └── csf_response.txt
                └── gm_response.txt
                └── wm_response.txt
                └── csf.nii.gz
                └── gm.nii.gz
                └── wmfod.nii.gz
                ├──fixel_directory
                    └── afd.nii.gz
                    └── peak_amp.nii.gz
                    └── index.nii.gz
                    └── directions.nii.gz
                    └── gt_threshold_fixels.nii.gz
                └── undersampled_fod
                    └── bvals
                    └── bvecs
                    └── data.nii.gz
                    └── normalised_data.nii.gz
                    └── csf_response.txt
                    └── gm_response.txt
                    └── wm_response.txt
                    └── csf.nii.gz
                    └── gm.nii.gz
                    └── wm.nii.gz
                └── tractseg (Only necessary for test subjects)    
```

Network options:
    The network options are set up in a hierarchical manner - the default options are set in the options.py script - it is not recommended that the user interacts with this file and instead uses the two other methods to change the network options. The first method is to use the comman line options - each option can be input to the train or test script as a command line option. A config file/path can also be input to the script as an option, which can then have the options formatted as a yaml file - see the documentation for more details. Due to the nature of the nesting any options specified in the config file will overwrite the options input via the command line, and the command line overwrites any options already in the options script. Note that the white space needs to be considered when inputting lists. 
