#!/bin/bash
inf_im=$1
subj=$2
save_dir=$3
rm -f tmp0.mif tmp1.mif tmp2.mif tmp_dir.mif tmp_afd.mif tmp_afd_abs.mif tmp_peak_amp_im.mif tmp_peak_amp_abs.mif tmp_peak_amp.mif

fod2fixel -afd afd.mif -peak_amp peak_amp.mif $inf_im $save_dir/fixel_directory

#Calculating the difference between the two fixel number images:
cp $save_dir/fixel_directory/index.mif tmp0.mif
mrconvert -force tmp0.mif -coord 3 0 tmp0.mif
mrconvert /media/duanj/F/joe/hcp_2/$subj/T1w/Diffusion/fixel_directory/index.mif -coord 3 0 tmp1.mif
mrcalc tmp0.mif tmp1.mif -sub tmp2.mif
mrcalc tmp2.mif -abs $save_dir/index_abs_err.mif

#Calculating the scalar fixel based analysis comparisons
fixel2voxel -number 11 $save_dir/fixel_directory/afd.mif none $save_dir/afd_im.mif
fixel2voxel -number 11 $save_dir/fixel_directory/peak_amp.mif none $save_dir/peak_amp_im.mif

mrcalc $save_dir/afd_im.mif /media/duanj/F/joe/hcp_2/$subj/T1w/Diffusion/fixel_directory/afd_im.mif -sub tmp_afd.mif 
mrcalc tmp_afd.mif -abs tmp_afd_abs.mif
mrmath tmp_afd_abs.mif sum $save_dir/afde.mif -axis 3 

mrcalc $save_dir/peak_amp_im.mif /media/duanj/F/joe/hcp_2/$subj/T1w/Diffusion/fixel_directory/peak_amp_im.mif -sub tmp_peak_amp.mif
mrcalc tmp_peak_amp.mif -abs tmp_peak_amp_abs.mif 
mrmath tmp_peak_amp_abs.mif sum $save_dir/pae.mif -axis 3

rm tmp0.mif tmp1.mif tmp2.mif tmp_dir.mif tmp_afd.mif tmp_afd_abs.mif tmp_peak_amp_im.mif tmp_peak_amp_abs.mif tmp_peak_amp.mif

#Writing the white matter pae to a text file
echo The average peak amplitude error over all the voxels in the white matter mask is:
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/pae.mif
echo $subj >> $save_dir/../wm_pae_stats.txt
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/pae.mif >> $save_dir/../wm_pae_stats.txt

#Writing the white matter afde to a text file
echo The average apparent fibre density error over all the voxels in the white matter mask is:
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/afde.mif
echo $subj >> $save_dir/../wm_afde_stats.txt
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/afde.mif >> $save_dir/../wm_afde_stats.txt

#Teh tract specifice FBA analysis (namely PAE and AFDE)
tract_seg_masks=/media/duanj/F/joe/hcp_2/$subj/T1w/Diffusion/tractseg/bundle_segmentations/

#Calculating the AFDE and PAE in the CC
echo Corpus Callosum >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/CC.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Corpus Callosum >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/CC.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt


#Calculating the AFDE and PAE in the MCP
echo Middle Cerebellar Penduncle >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/MCP.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Middle Cerebellar Penduncle >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/MCP.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt

#Calculating the AFDE and PAE in the CST
echo Cerebrospinal Tract >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/CST_whole.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Cerebrospinal Tract >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/CST_whole.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt

#Calculating the AFDE and PAE in the CC and CC with only 1 fixel
echo Corpus Callosum Single Fixel >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/CC_1fixel.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Corpus Callosum Single Fixel >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/CC_1fixel.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt

#Calculating the AFDE and PAE in the MCP crossing with the CST with only 2 fixels 
echo Middle Cerebellar Penduncle and Cerebrospinal Tract Crossing with Two Fixels >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/MCP_CST_2fixel.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Middle Cerebellar Penduncle and Cerebrospinal Tract Crossing with Two Fixels >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/MCP_CST_2fixel.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt

#Calculating the AFDE and PAE in the CC,CST and SLF with only 3 fibres
echo Corpus Callosum, Cerebrospinal Tract and the Superior Longitudinal Fascicle Three Fixel >> $save_dir/tract_specific_afde.txt
mrstats -mask $tract_seg_masks/CC_CST_SLF_3fixel.nii.gz $save_dir/afde.mif >> $save_dir/tract_specific_afde.txt
echo >> $save_dir/tract_specific_afde.txt
echo Corpus Callosum, Cerebrospinal Tract and the Superior Longitudinal Fascicle Three Fixel >> $save_dir/tract_specific_pae.txt
mrstats -mask $tract_seg_masks/CC_CST_SLF_3fixel.nii.gz $save_dir/pae.mif >> $save_dir/tract_specific_pae.txt
echo >> $save_dir/tract_specific_pae.txt
