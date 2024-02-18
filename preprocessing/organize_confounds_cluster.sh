#!/bin/bash
# remove headers from each run


declare -a file_list
declare -a confound_list

SUBJ=${SLURM_ARRAY_TASK_ID}

# copy confounds files to project files


confound_list=("${confound_list[@]}" "/work/abslab/AVFP/fmriprep/$SUBJ/fmriprep/sub-$SUBJ/func")


for confound_file in "${confound_list[@]}"
	do
	echo copying $confound_file
	cp $confound_file/*task-affvids_run-*_desc-confounds_timeseries.tsv /scratch/wang.yiyu/AVFP/confounds
	cp $confound_file/*task-socialpred_run-*_desc-confounds_timeseries.tsv /scratch/wang.yiyu/SocialAbstraction/confounds
done
