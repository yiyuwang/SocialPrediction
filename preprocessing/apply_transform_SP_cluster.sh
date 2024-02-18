#!/bin/bash

REF_2MM="/scratch/wang.yiyu/AVFP/masks/MNI152_T1_2mm_brain.nii.gz"
REF_3MM="/scratch/wang.yiyu/AVFP/masks/FSL_MNI152_T1_3mm_brain.nii.gz"


# register to 3mm

mkdir "/scratch/wang.yiyu/SocialAbstraction/transformed_data_3mm/"
mkdir "/scratch/wang.yiyu/SocialAbstraction/transform_info_3mm"

subjID=${SLURM_ARRAY_TASK_ID}


task="socialpred"
for r in 1 2
    do
        echo run $r
        FUNC_VOL="/work/abslab/AVFP/fmriprep/${subjID}/fmriprep/sub-${subjID}/func/sub-${subjID}_task-${task}_run-${r}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        highres="/work/abslab/AVFP/fmriprep/${subjID}/fmriprep/sub-${subjID}/anat/sub-${subjID}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"
        REF_MAT_F2S="/scratch/wang.yiyu/SocialAbstraction/transform_info_3mm/sub-${subjID}_${task}_run${r}_func2highres.mat"
        REF_MAT_2MM_TO_3MM="/scratch/wang.yiyu/SocialAbstraction/transform_info_3mm/sub-${subjID}_${task}_run${r}_highres2standard_3mm.mat"
        REF_MAT_F_TO_3MM="/scratch/wang.yiyu/SocialAbstraction/transform_info_3mm/sub-${subjID}_${task}_run${r}_func2standard_3mm.mat"
        VOL_3MM="/scratch/wang.yiyu/SocialAbstraction/transformed_data_3mm/sub-${subjID}_${task}_run${r}.nii.gz"
        
        # register func to highres
        flirt -ref $highres -in $FUNC_VOL -omat $REF_MAT_F2S

        # register highres to 3mm standard
        flirt -ref $REF_3MM -in $highres -omat $REF_MAT_2MM_TO_3MM
    
        # combine transforms: convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
        convert_xfm -omat $REF_MAT_F_TO_3MM -concat $REF_MAT_2MM_TO_3MM $REF_MAT_F2S

        # register func to 3mm standard
        flirt -in $FUNC_VOL -ref $REF_3MM -out $VOL_3MM -init $REF_MAT_F_TO_3MM -applyxfm
done


# task="socialpred"


echo Running Subj $subjID
mkdir "/scratch/wang.yiyu/SocialAbstraction/transformed_data_2mm/"
mkdir "/scratch/wang.yiyu/SocialAbstraction/transform_info_2mm"

for r in 1 2
    do
    echo run $r

    FUNC_VOL="/work/abslab/AVFP/fmriprep/${subjID}/fmriprep/sub-${subjID}/func/sub-${subjID}_task-${task}_run-${r}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    highres="/work/abslab/AVFP/fmriprep/${subjID}/fmriprep/sub-${subjID}/anat/sub-${subjID}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"
    REF_MAT_F2S="/scratch/wang.yiyu/SocialAbstraction/transform_info_2mm/sub-${subjID}_${task}_run${r}_func2highres.mat"
    REF_MAT_2MM="/scratch/wang.yiyu/SocialAbstraction/transform_info_2mm/sub-${subjID}_${task}_run${r}_highres2standard_2mm.mat"
    REF_MAT_F_TO_2MM="/scratch/wang.yiyu/SocialAbstraction/transform_info_2mm/sub-${subjID}_${task}_run${r}_func2standard_2mm.mat"
    VOL_2MM="/scratch/wang.yiyu/SocialAbstraction/transformed_data_2mm/sub-${subjID}_${task}_run${r}.nii.gz"

    # register func to highres
    flirt -ref $highres -in $FUNC_VOL -omat $REF_MAT_F2S

    # register highres to 2mm standard
    flirt -ref $REF_2MM -in $highres -omat $REF_MAT_2MM

    # combine transforms: convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
    convert_xfm -omat $REF_MAT_F_TO_2MM -concat $REF_MAT_2MM $REF_MAT_F2S

    # register func to 3mm standard
    flirt -in $FUNC_VOL -ref $REF_2MM -out $VOL_2MM -init $REF_MAT_F_TO_2MM -applyxfm
done


