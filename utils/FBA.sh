#!/bin/bash
inf_im=$1
subj=$2
save_dir=$3

fod2fixel -afd afd.mif -peak_amp peak_amp.mif $inf_im $save_dir/fixel_directory

#Calculating the difference between the two fixel number images:
cp $save_dir/fixel_directory/index.mif tmp0.mif
mrconvert tmp0.mif -coord 3 0 tmp0.mif
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

#Writing the whole brain pae to a text file
echo The average peak amplitude error over all the voxels in the whole brain is:
mrstats $save_dir/pae.mif
echo $subj >> $save_dir/../wb_pae_stats.txt
mrstats $save_dir/pae.mif >> $save_dir/../wb_pae_stats.txt

#Writing the white matter afde to a text file
echo The average apparent fibre density error over all the voxels in the white matter mask is:
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/afde.mif
echo $subj >> $save_dir/../wm_afde_stats.txt
mrstats -mask /media/duanj/F/joe/hcp_2/$subj/T1w/white_matter_mask.mif $save_dir/afde.mif >> $save_dir/../wm_afde_stats.txt

#Writing the whole brain afde to a text file
echo The average apparent fibre density error over all the voxels in the whole brain is:
mrstats $save_dir/afde.mif
echo $subj >> $save_dir/../wb_afde_stats.txt
mrstats >> $save_dir/../wb_afde_stats.txt
