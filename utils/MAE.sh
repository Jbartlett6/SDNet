#!/bin/bash

inf_im=$1
subj=$2
save_dir=$3
data_path=$4

fixel2peaks -number 11 $save_dir/fixel_directory $save_dir/fixel_directory/fixel_directions.nii.gz

python utils/MAE.py --inference_image $inf_im --subject $subj --save_directory $save_dir --data_path $data_path

mrcalc mae_tmp.nii.gz -abs mae_tmp_pos.nii.gz
mrcalc mae_tmp_pos.nii.gz -acos $save_dir/mae.nii.gz


mrthreshold -abs 0.1 -comparison gt mae_tmp_pos.nii.gz wb_mask_tmp.nii.gz


#Writing the white matter mae to a text file
echo The average mean angular error over all the voxels in the white matter mask is:
mrstats -allvolumes -mask $data_path/$subj/T1w/white_matter_mask.mif $save_dir/mae.nii.gz
echo $subj >> $save_dir/../wm_mae_stats.txt
mrstats -allvolumes -mask $data_path/$subj/T1w/white_matter_mask.mif $save_dir/mae.nii.gz >> $save_dir/../wm_mae_stats.txt

#Writing the white matter pae to a text file
echo The average mean angular error over all the voxels in the white matter mask is:
mrstats -allvolumes -mask wb_mask_tmp.nii.gz $save_dir/mae.nii.gz 
echo $subj >> $save_dir/../wb_mae_stats.txt
mrstats -allvolumes -mask wb_mask_tmp.nii.gz $save_dir/mae.nii.gz>> $save_dir/../wb_mae_stats.txt 

rm mae_tmp.nii.gz mae_tmp_pos.nii.gz wb_mask_tmp.nii.gz