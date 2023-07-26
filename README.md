# SDNet 
The code is this repository implements SDNet, a model-driven FOD reconstruction network, for further details see the accompanying paper at . The code in this repo is currently being updated to improve usability. 

## Abstract
Fibre orientation distribution (FOD) reconstruction using deep learning has the potential to produce accurate FODs from a reduced number of diffusion-weighted images (DWIs), decreasing total imaging time. Diffusion acquisition invariant representations of the DWI signals are typically used as input to these methods to ensure that they can be applied flexibly to data with different b-vectors and b-values; however, this means the network cannot condition its output directly on the DWI signal. In this work, we propose a spherical deconvolution network, a model-driven deep learning FOD reconstruction architecture, that ensures intermediate and output FODs produced by the network are consistent with the input DWI signals. Furthermore, we implement a fixel classification penalty within our loss function, encouraging the network to produce FODs that can subsequently be segmented into the correct number of fixels and improve downstream fixel-based analysis. Our results show that the model-based deep learning architecture achieves competitive performance compared to a state-of-the-art FOD super-resolution network, FOD-Net. Moreover, we show that the fixel classification penalty can be tuned to offer improved performance with respect to metrics that rely on accurately segmented of FODs.

<img src="SDNet%20(2).jpg" width="1000"><br>
Figure 1 **SDNet achitecture**

# User Guide 
## File Structure
The directory the code comes in is named CSDNet_dir, and will have the following structure:
```bash
SDNet_dir
    └── data.py
    └── util.py
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
The network is designed to be trained on the WU-MINN human connectome project dataset, as a consequence the data directory is designed to be similar to the HCP data when downloaded. The main data directory, named hcp in this case, subjects labeled as per the HCP. The file path of this directory should be specified in options.py under the attribute `data_dir`. To specify the subjects in `data_dir` to be used for training, validation and testing adjust the `train_subject_list`, `val_subject_list`, and `test_subject_list` attributes.

```bash
.
└── hcp
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
                └── tractseg *
                    └── peaks.nii.gz *
                    └── bundle_segmentations *
                        └── ** bundle_segmnetation_masks **
                    
```

## Network Training

## Network Inference

## Network Configuration:
    The network options are set up in a hierarchical manner - the default options are set in the options.py script - it is not recommended that the user interacts with this file and instead uses the two other methods to change the network options. The first method is to use the comman line options - each option can be input to the train or test script as a command line option. A config file/path can also be input to the script as an option, which can then have the options formatted as a yaml file - see the documentation for more details. Due to the nature of the nesting any options specified in the config file will overwrite the options input via the command line, and the command line overwrites any options already in the options script. Note that the white space needs to be considered when inputting lists. 
