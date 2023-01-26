#!/bin/bash
rm -f mult_coeff.mif num.mif sum_1.mif sum_2.mif denom.mif tmp_1.mif tmp_2.mif 
save_path=$3
mrconvert $1 -coord 3 1:44 -axes 0,1,2,3 tmp_1.mif
mrconvert $2 -coord 3 1:44 -axes 0,1,2,3 tmp_2.mif
subject=$4


mrcalc tmp_1.mif tmp_2.mif -mult mult_coeff.mif
mrmath mult_coeff.mif sum num.mif -axis 3
mrmath tmp_1.mif norm sum_1.mif -axis 3
mrmath tmp_2.mif norm sum_2.mif -axis 3
mrcalc sum_1.mif sum_2.mif -multiply denom.mif
rm out_inference.mif
mrcalc num.mif denom.mif -divide $save_path/acc_tmp.mif

rm mult_coeff.mif num.mif sum_1.mif sum_2.mif denom.mif tmp_1.mif tmp_2.mif 

echo 'The stats for the ACC in the white matter are:'
mrstats -mask /media/duanj/F/joe/hcp_2/$subject/T1w/white_matter_mask.mif $save_path/acc_tmp.mif
echo $subject wm: >> $save_path/../wm_acc_stats.txt 
mrstats -mask /media/duanj/F/joe/hcp_2/$subject/T1w/white_matter_mask.mif $save_path/acc_tmp.mif >> $save_path/../wm_acc_stats.txt

echo 'The stats for the ACC over the whole brain are:'
mrstats $save_path/acc_tmp.mif
echo $subject wb: >> $save_path/../wb_acc_stats.txt 
mrstats  $save_path/acc_tmp.mif >> $save_path/../wb_acc_stats.txt

