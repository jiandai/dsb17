'''
Naive or even ``non-sense'' CNN

ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
ver 60170312 by jian: run on server, 128*512*512 kills tf on server
ver 60170313.1 by jian: 1st pass of all cases
ver 60170313.2 by jian: consolidate preproc, check the xy-size, orientation
ver 60170313.3 by jian: resample and save to npz
ver 60170313.4 by jian: split the script, and this portion becomes "m0-preproc.py"

to-do: 
=> pre-proc test set
=> npz for mini batch in parallel
'''


#N=1500 # more than the number of cases
N=2

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
#test_csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id')


import os
import numpy as np
import scipy.ndimage
import dicom


patients = labels_csv.iloc[:N,:].index
#patients = test_csv.iloc[:N,:].index

from dicom_batch import get_one_scan
vox_list=[]
log=[]
images_path = '../input/stage1/'
for i,pat in enumerate(patients):
	img = get_one_scan(images_path+pat,resampling=True)
	log.append([i,pat,img.shape[0],img.shape[1],img.shape[2]])
	vox_list.append(img)



print len(vox_list)
logDF=pd.DataFrame(log,columns=['index','id','sz','sx','sy'])
print logDF
#logDF.to_csv("preproc-log.csv")
#npz_path = '../process/prep-out/resampled-out.npz'
#np.savez_compressed(npz_path,vox_list)


