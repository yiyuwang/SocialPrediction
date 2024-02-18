#!/bin/bash

#SBATCH --job-name=2_fmriprep                 # sets the job name
#SBATCH --nodes=1                                 # reserves 1 machines
#SBATCH --tasks-per-node=1                        # sets 1 tasks for each machine
#SBATCH --cpus-per-task=8                         # sets 8 core for each task
#SBATCH --mem=50Gb                               # reserves 50 GB memory
#SBATCH --partition=short                 	  	  # requests that the job is executed in partition short
#SBATCH --time=23:59:00                           # reserves machines/cores for 20 hours.
#SBATCH --output=2_fmriprep.%A-%a.out               # sets the standard output to be stored in file %A-%a - Array job id (A) and task id (a))
#SBATCH --error=2_fmriprep.%A-%a.err                # sets the standard error to be stored in file %A-%a - Array job id (A) and task id (a))
#SBATCH --array=167

module load python/3.7.1
srun python Edit_fmap_json.py ${SLURM_ARRAY_TASK_ID}

module load singularity
srun run_fmriprep_fmap_reg.sh ${SLURM_ARRAY_TASK_ID}

module load fsl/6.0.0
srun organize_confounds_cluster_SP.sh ${SLURM_ARRAY_TASK_ID}
srun apply_transform_SP_cluster.sh ${SLURM_ARRAY_TASK_ID}


