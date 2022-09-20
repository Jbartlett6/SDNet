#!/bin/bash

inf_im=$1
subj=$2
save_dir=$3
data_path=$4

fixel2peaks -number 11 $save_dir/fixel_directory $save_dir/fixel_directory/fixel_directions.nii.gz

python utils/MAE.py $inf_im $subj $save_dir $data_path

mrcalc mae_tmp.nii.gz -acos true_mae_tmp.nii.gz

#Writing the white matter pae to a text file
echo The average mean angular error over all the voxels in the white matter mask is:
mrstats -allvolumes -ignorezero -mask $data_path/$subj/T1w/white_matter_mask.mif true_mae_tmp.nii.gz
echo $subj >> $save_dir/../wm_mae_stats.txt
mrstats -allvolumes -ignorezero -mask $data_path/$subj/T1w/white_matter_mask.mif true_mae_tmp.nii.gz >> $save_dir/../wm_mae_stats.txt

#Writing the white matter pae to a text file
echo The average mean angular error over all the voxels in the white matter mask is:
mrstats -allvolumes -ignorezero true_mae_tmp.nii.gz 
echo $subj >> $save_dir/../wb_mae_stats.txt
mrstats -allvolumes -ignorezero true_mae_tmp.nii.gz >> $save_dir/../wb_mae_stats.txt 

rm mae_tmp.nii.gz true_mae_tmp.nii.gz