'''
Naive or even ``non-sense'' CNN

ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
ver 60170312 by jian: run on server, 128*512*512 kills tf on server
ver 60170313.1 by jian: 1st pass of all cases
ver 60170313.2 by jian: consolidate preproc, check the xy-size, orientation
ver 60170313.3 by jian: resample and save to npz
ver 60170313.4 by jian: split the script, and this portion becomes "m0-preproc.py"
ver 60170314.1 by jian: test save individual npz /w test set
ver 60170314.2 by jian: test save batch by using LSF array /w test set
ver 60170314.2 by jian: save batch by using LSF array /w tr set

'''
import sys
# assume len(sys.argv)>1:
# assertion: input an integer
BATCH_INDEX = int(sys.argv[1])
#BATCH_SIZE =25 # for test set /w 8 batches for 198 records
BATCH_SIZE = 35 # for training set /w 40 batches for 1397 records
#BATCH_SIZE =2 
batch_range = range((BATCH_INDEX-1)*BATCH_SIZE,BATCH_INDEX*BATCH_SIZE)
batch_start = (BATCH_INDEX-1)*BATCH_SIZE
batch_end = BATCH_INDEX*BATCH_SIZE



#N=1500 # more than the number of cases

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
#test_csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id')


import os
import numpy as np
import scipy.ndimage
import dicom


patients = labels_csv.index[batch_start:batch_end]
#patients = test_csv.index[batch_start:batch_end]
print patients
print batch_range
print len(patients)

from dicom_batch import get_one_scan
vox_list=[]
log=[]
images_path = '../input/stage1/'
#output_path = '../process/prep-out/test/'
output_path = '../process/prep-out/training/'
for i,pat in enumerate(patients):
	img = get_one_scan(images_path+pat,resampling=True)
	log.append([i,pat,img.shape[0],img.shape[1],img.shape[2]])
	vox_list.append(img)
	# indiviudal scan saving
	#npz_path = output_path+pat+'.npz'
	#np.savez_compressed(npz_path,img)



print len(vox_list)
logDF=pd.DataFrame(log,columns=['index','id','sz','sx','sy'])
print logDF
#logDF.to_csv("preproc-test-set-batch-"+sys.argv[1]+"-log.csv")
logDF.to_csv("preproc-training-set-batch-"+sys.argv[1]+"-log.csv")
#logDF.to_csv("preproc-log.csv")
#npz_file = "preproc-test-set-batch-"+sys.argv[1]+".npz"
npz_file = "preproc-training-set-batch-"+sys.argv[1]+".npz"
print npz_file
npz_path = output_path + npz_file
np.savez_compressed(npz_path,vox_list)


