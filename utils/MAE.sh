#!/bin/bash

inf_im=$1
subj=$2
save_dir=$3
data_path=$4

fixel2peaks -number 11 $save_dir/fixel_directory $save_dir/fixel_directory/fixel_directions.nii.gz

python utils/MAE.py --inference_image $inf_im --subject $subj --save_directory $save_dir --data_path $data_path

mrcalc mae_tmp.nii.gz -abs mae_tmp_pos.nii.gz
mrcalc mae_tmp_pos.nii.gz -acos $save_dir/mae.nii.gz
mrconvert -coord 3 0 $save_dir/mae.nii.gz $save_dir/mae_fix_1.nii.gz
mrconvert -coord 3 1 $save_dir/mae.nii.gz $save_dir/mae_fix_2.nii.gz
mrconvert -coord 3 2 $save_dir/mae.nii.gz $save_dir/mae_fix_3.nii.gz 


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

#The tract specifice FBA analysis (namely PAE and AFDE)
tract_seg_masks=/media/duanj/F/joe/hcp_2/$subj/T1w/Diffusion/tractseg/bundle_segmentations/

#Removing previous results if they already exist to prevent them containing repeat results:
rm -f $save_dir/tract_specific_mae_fix_1.txt $save_dir/tract_specific_mae_fix_2.txt $save_dir/tract_specific_mae_fix_3.txt

#MAE in CC
echo Corpus Callosum >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/CC.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Corpus Callosum >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/CC.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Corpus Callosum >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/CC.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt

#MAE in MCP
echo Middle Cerebellar Penduncle >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/MCP.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Middle Cerebellar Penduncle >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/MCP.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Middle Cerebellar Penduncle >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/MCP.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt

#MAE in CST
echo Cerebrospinal Tract >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/CST_whole.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Cerebrospinal Tract >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/CST_whole.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Cerebrospinal Tract >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/CST_whole.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt

#Calculating the in the CC with only 1 fixel
echo Corpus Callosum Single Fixel >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/CC_1fixel.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Corpus Callosum Single Fixel >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/CC_1fixel.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Corpus Callosum Single Fixel >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/CC_1fixel.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt

#Calculating the MAE in the MCP crossing with the CST with only 2 fixels 
echo Middle Cerebellar Penduncle and Cerebrospinal Tract Crossing with Two Fixels >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/MCP_CST_2fixel.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Middle Cerebellar Penduncle and Cerebrospinal Tract Crossing with Two Fixels >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/MCP_CST_2fixel.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Middle Cerebellar Penduncle and Cerebrospinal Tract Crossing with Two Fixels >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/MCP_CST_2fixel.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt

#Calculating the MAE in the crossing between the CC, CST and SLF with only 3 fixels. 
echo Corpus Callosum, Cerebrospinal Tract and the Superior Longitudinal Fascicle Three Fixels >> $save_dir/tract_specific_mae_fix_1.txt
mrstats -mask $tract_seg_masks/CC_CST_SLF_3fixel.nii.gz $save_dir/mae_fix_1.nii.gz >> $save_dir/tract_specific_mae_fix_1.txt
echo >> $save_dir/tract_specific_mae_fix_1.txt

echo Corpus Callosum, Cerebrospinal Tract and the Superior Longitudinal Fascicle Three Fixels >> $save_dir/tract_specific_mae_fix_2.txt
mrstats -mask $tract_seg_masks/CC_CST_SLF_3fixel.nii.gz $save_dir/mae_fix_2.nii.gz >> $save_dir/tract_specific_mae_fix_2.txt
echo >> $save_dir/tract_specific_mae_fix_2.txt

echo Corpus Callosum, Cerebrospinal Tract and the Superior Longitudinal Fascicle Three Fixels >> $save_dir/tract_specific_mae_fix_3.txt
mrstats -mask $tract_seg_masks/CC_CST_SLF_3fixel.nii.gz $save_dir/mae_fix_3.nii.gz >> $save_dir/tract_specific_mae_fix_3.txt
echo >> $save_dir/tract_specific_mae_fix_3.txt










rm mae_tmp.nii.gz mae_tmp_pos.nii.gz wb_mask_tmp.nii.gz