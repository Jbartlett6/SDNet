#!/bin/bash
model_dir=/media/duanj/F/joe/CSD_experiments/FODNet
rm $model_dir/inference/sse_stats.txt
while read -r subject;
do
echo $subject
#Inference
inf_dir=$model_dir/inference/$subject
inf_fod_dir=$inf_dir/inf_wm_fod.nii.gz

tract_seg_masks=/media/duanj/F/joe/hcp_2/$subject/T1w/Diffusion/tractseg/bundle_segmentations/
gt_fod_dir=/media/duanj/F/joe/hcp_2/$subject/T1w/Diffusion/wmfod.nii.gz

rm -f $inf_dir/mse.nii.gz $inf_dir/sse.nii.gz 
mrcalc $inf_fod_dir $gt_fod_dir -sub 2 -pow $inf_dir/sq_dif.nii.gz
mrmath $inf_dir/sq_dif.nii.gz sum $inf_dir/sse.nii.gz -axis 3
rm $inf_dir/sq_dif.nii.gz

#Masks
wm_mask=/media/duanj/F/joe/hcp_2/$subject/T1w/white_matter_mask.mif
CC_mask=$tract_seg_masks/CC.nii.gz
MCP_mask=$tract_seg_masks/MCP.nii.gz
CST_mask=$tract_seg_masks/CST_whole.nii.gz
CC_1fix=$tract_seg_masks/CC_1fixel.nii.gz
twofix_mask=$tract_seg_masks/MCP_CST_2fixel.nii.gz
threefix_mask=$tract_seg_masks/CC_CST_SLF_3fixel.nii.gz

rm $inf_dir/tract_specific_SSE_stats.txt

#Writing the white matter stats:
echo $subject >> $inf_dir/../sse_stats.txt
mrstats -mask $wm_mask $inf_dir/sse.nii.gz >> $inf_dir/../sse_stats.txt 

#Writing the anatomy stats:
echo Corpus Callosum >> $inf_dir/tract_specific_SSE_stats.txt 
mrstats -mask $CC_mask $inf_dir/sse.nii.gz >> $inf_dir/tract_specific_SSE_stats.txt 

#Writing the anatomy stats:
echo MCP >> $inf_dir/tract_specific_SSE_stats.txt 
mrstats -mask $MCP_mask $inf_dir/sse.nii.gz >> $inf_dir/tract_specific_SSE_stats.txt

#Writing the anatomy stats:
echo Cerebrospinal Tract >> $inf_dir/tract_specific_SSE_stats.txt 
mrstats -mask $CST_mask $inf_dir/sse.nii.gz >> $inf_dir/tract_specific_SSE_stats.txt 

echo Corpus Callosum 1 fixel >> $inf_dir/tract_specific_SSE_stats.txt 
mrstats -mask $CC_1fix $inf_dir/sse.nii.gz >> $inf_dir/tract_specific_SSE_stats.txt 


#Calculating the sse for the subject
done < "/home/jxb1336/code/Project_1: HARDI_Recon/FOD-REG_NET/CSDNet_dir/utils/validation_subjects.txt"
