#!/bin/bash


subj=${SLURM_ARRAY_TASK_ID}

# for subj in "${subjNo[@]}"
	
# 	do
# 	mkdir /scratch/wang.yiyu/AVFP/fmriprep/$subj
# 	mkdir /scratch/wang.yiyu/AVFP/work/$subj


# 	singularity run \
# 	-B /home/wang.yiyu/.cache/fmriprep:/home/fmriprep/.cache/fmriprep \
# 	--bind /scratch/wang.yiyu \
# 	--cleanenv /scratch/wang.yiyu/fmriprep-1.5.4.simg \
# 	/scratch/wang.yiyu/AVFP/BIDS/ /scratch/wang.yiyu/AVFP/fmriprep/ \
# 	participant \
# 	--participant-label sub-$subj \
# 	-w /scratch/wang.yiyu/AVFP/work/$subj \
# 	--nthreads 8 \
# 	--low-mem \
# 	--fs-license-file /scratch/wang.yiyu/AVFP/misc/freesurfer.txt \
# 	--fs-no-reconall \
# 	--use-plugin /scratch/wang.yiyu/AVFP/misc/workaround.yml \
# 	--output-spaces MNI152NLin6Asym:res-2 anat 
# done


# use job array id as sub id such that run all participants in parallel
singularity run \
-B /home/wang.yiyu/.cache/fmriprep:/home/fmriprep/.cache/fmriprep \
--bind /work/abslab/AVFP \
--bind /scratch/wang.yiyu \
--cleanenv /scratch/wang.yiyu/fmriprep-20.2.1.simg \
/work/abslab/AVFP/BIDS/ /work/abslab/AVFP/fmriprep/$subj \
participant \
--participant-label sub-$subj \
-w /scratch/wang.yiyu/AVFP/work/$subj \
--nthreads 8 \
--output-spaces MNI152NLin6Asym:res-2 T1w \
--fd-spike-threshold 0.6 --dvars-spike-threshold 2 \
--fs-no-reconall \
--fs-license-file /scratch/wang.yiyu/AVFP/misc/freesurfer.txt