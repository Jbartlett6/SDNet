#!/bin/bash
data_path=$1
#folder_name=$2

#subjectlist=hcp-folders-formatted.txt
#subjectlist=/media/duanj/F/joe/data/hcp_metadata/hcp_test_subjects.txt
traininglist=train_subjects.txt
#vallist=validation_subjects.txt
# traininglist=test.txt
vallist=validation_subjects.txt


while read -r subject;
do
path=$data_path/$subject/T1w/Diffusion
5ttgen fsl $path/../T1w_acpc_dc_restore_1.25.nii.gz $path/../5ttgen.nii.gz -nocrop

done < $traininglist