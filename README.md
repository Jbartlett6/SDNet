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

