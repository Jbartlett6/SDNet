#!/bin/bash
data_path=$1
folder_name=$2

#subjectlist=hcp-folders-formatted.txt
#subjectlist=/media/duanj/F/joe/data/hcp_metadata/hcp_test_subjects.txt
traininglist=train_subjects.txt
vallist=validation_subjects.txt
while read -r subject;
do
	echo Preparing the training data.
	path=$data_path/$subject/T1w/Diffusion
	echo $subject
	echo $path
    mkdir $data_path/$subject/T1w/Diffusion/$folder_name
	
    #Calculating the fully sampled response function and the fully sampled fod
	dwi2response dhollander $path/data.nii.gz $path/wm_response.txt $path/gm_response.txt $path/csf_response.txt -fslgrad $path/bvecs $path/bvals
	dwi2fod -fslgrad $path/bvecs $path/bvals msmt_csd $path/data.nii.gz $path/wm_response.txt $path/wmfod.nii.gz $path/gm_response.txt $path/gm.nii.gz $path/csf_response.txt $path/csf.nii.gz 	
	mrconvert $path/gt_fod.nii.gz -coord 3 0:44 $path/gt_wmfod.nii.gz

    #Undersampling the data.
    python dwi_undersample.py $data_path $subject $folder_name

	#Normalising the already undersampled data so the maximum value is 1.
	dwinormalise individual $path/$folder_name/data.nii.gz $path/nodif_brain_mask.nii.gz $path/$folder_name/normalised_data.nii.gz -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals -intensity 1

	#Calculating the undersampled response functions to be input into the network:
	dwi2response dhollander $path/$folder_name/normalised_data.nii.gz $path/$folder_name/wm_response.txt $path/$folder_name/gm_response.txt $path/$folder_name/csf_response.txt -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals 
	dwi2fod -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals msmt_csd $path/$folder_name/data.nii.gz $path/$folder_name/wm_response.txt $path/$folder_name/undersampled_wm_fod.nii.gz $path/$folder_name/gm_response.txt $path/$folder_name/undersampled_gm_fod.nii.gz $path/$folder_name/csf_response.txt $path/$folder_name/undersampled_csf_fod.nii.gz
	mrcat -axis 3 $path/$folder_name/undersampled_wm_fod.nii.gz $path/$folder_name/undersampled_gm_fod.nii.gz $path/$folder_name/undersampled_csf_fod.nii.gz $path/$folder_name/$folder_name.nii.gz
	mrcat -axis 3 $path/wmfod.nii.gz $path/gm.nii.gz $path/csf.nii.gz $path/$folder_name/gt_fod.nii.gz	
	cp $path/wmfod.nii.gz $path/$folder_name/gt_wm_fod.nii.gz
	
	
done < $traininglist

while read -r subject;
do
	echo Preparing Validation subejects
	path=$data_path/$subject/T1w/Diffusion
	echo $subject
	echo $path
    mkdir $data_path/$subject/T1w/Diffusion/$folder_name

	#Calculating the fully sampled response function and the fully sampled fod
	dwi2response dhollander $path/data.nii.gz $path/wm_response.txt $path/gm_response.txt $path/csf_response.txt -fslgrad $path/bvecs $path/bvals
	dwi2fod -fslgrad $path/bvecs $path/bvals msmt_csd $path/data.nii.gz $path/wm_response.txt $path/wmfod.nii.gz $path/gm_response.txt $path/gm.nii.gz $path/csf_response.txt $path/csf.nii.gz 	
	mrconvert $path/gt_fod.nii.gz -coord 3 0:44 $path/gt_wmfod.nii.gz

	#Calculating the 5ttgen fsl mask. 
	5ttgen fsl $path/../T1w_acpc_dc_restore_1.25.nii.gz $path/../5ttgen.mif -nocrop
	mrconvert $path/../5ttgen.mif -coord 3 2 $path/../white_matter_mask.mif

	#Carrying out fod2fixel for gt fixels.
	fod2fixel -afd afd.mif -peak_amp peak_amp.mif $path/wmfod.nii.gz $path/fixel_directory
	fixel2voxel -number 11 $path/fixel_directory/peak_amp.mif none $path/fixel_directory/peak_amp_im.mif
	fixel2voxel -number 11 $path/fixel_directory/afd.mif none $path/fixel_directory/afd_im.mif

	#Undersampling the data:
	python dwi_undersample.py $data_path $subject $folder_name
	
	#Normalising the already undersampled data so the maximum value is 1.
	dwinormalise individual $path/$folder_name/data.nii.gz $path/nodif_brain_mask.nii.gz $path/$folder_name/normalised_data.nii.gz -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals -intensity 1
	
	#Calculating the undersampled response functions and FOD to be input into the network:
	dwi2response dhollander $path/$folder_name/normalised_data.nii.gz $path/$folder_name/wm_response.txt $path/$folder_name/gm_response.txt $path/$folder_name/csf_response.txt -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals 
	dwi2fod -fslgrad $path/$folder_name/bvecs $path/$folder_name/bvals msmt_csd $path/$folder_name/data.nii.gz $path/$folder_name/wm_response.txt $path/$folder_name/undersampled_wm_fod.nii.gz $path/$folder_name/gm_response.txt $path/$folder_name/undersampled_gm_fod.nii.gz $path/$folder_name/csf_response.txt $path/$folder_name/undersampled_csf_fod.nii.gz
	mrcat -axis 3 $path/$folder_name/undersampled_wm_fod.nii.gz $path/$folder_name/undersampled_gm_fod.nii.gz $path/$folder_name/undersampled_csf_fod.nii.gz $path/$folder_name/$folder_name.nii.gz
	mrcat -axis 3 $path/wmfod.nii.gz $path/gm.nii.gz $path/csf.nii.gz $path/$folder_name/gt_fod.nii.gz	
	cp $path/wmfod.nii.gz $path/$folder_name/gt_wm_fod.nii.gz


done < $vallist
