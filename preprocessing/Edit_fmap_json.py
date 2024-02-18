#!/usr/bin/env python
# coding: utf-8


import os
import json
import sys
import glob

#subjectNo = [100] + list(range(103,153))
subjectNo = [100]
#BIDS_dir = '/Users/yiyuwang/ImgagingData/AVFP/BIDS'
BIDS_dir = '/work/abslab/AVFP/BIDS/'


array_id_position = 1


job_array_id = sys.argv[array_id_position]
print ("Job_array_id %i: %s" % (array_id_position, job_array_id))



sub_num = job_array_id
print ("subject number: ", sub_num)

subject = 'sub-' + sub_num


if os.path.exists(BIDS_dir + subject):
        func = os.listdir(BIDS_dir + subject + '/func')
        fmap = glob.glob(BIDS_dir + subject + '/fmap/*json')

    # read the orignal json file first
    # specify the intended func run names
    # append the intended to the end of the json file
    # resave the json file


        for file in fmap:
            with open(file) as f:
                json_data = json.load(f)

            if 'run-01_epi' in file:     
                # grab the file names for the intended nifti files:
                files = [i for i in func if 'task-affvids_run-01_bold.nii.gz' in i or 'task-affvids_run-02_bold.nii.gz' in i]


            elif 'run-02_epi' in file:     
                # grab the file names for the intended nifti files:
                files = [i for i in func if 'task-affvids_run-03_bold.nii.gz' in i or 'task-affvids_run-04_bold.nii.gz' in i or 'task-affvids_run-05_bold.nii.gz' in i]



            elif 'run-03_epi' in file:     
                # grab the file names for the intended nifti files:
                files = [i for i in func if '.nii.gz' in i and 'task-socialpred' in i]


            funcRuns = ['func/' + s for s in files]   
            json_data["IntendedFor"]=funcRuns
            os.chmod(file, 0o664)
            with open(file,'w') as json_file:
                json.dump(json_data, json_file, indent=4)

    