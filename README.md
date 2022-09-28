# CSDNet
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
The directory structure of the data directory is as follows. The data directory contains all of the subjects which you have access to. Which of these subjects are used for training, validation and testing is then determined by lists which are input into the script/dataset

```bash
.
└── data_directory
    └── subject
        └── T1w
            └── Diffusion
                ├──fixel_directory
                └── sub_dir 
```

Network options:
    The network options are set up in a hierarchical manner - the default options are set in the options.py script - it is not recommended that the user interacts with this file and instead uses the two other methods to change the network options. The first method is to use the comman line options - each option can be input to the train or test script as a command line option. A config file/path can also be input to the script as an option, which can then have the options formatted as a yaml file - see the documentation for more details. Due to the nature of the nesting any options specified in the config file will overwrite the options input via the command line, and the command line overwrites any options already in the options script. Note that the white space needs to be considered when inputting lists. 
