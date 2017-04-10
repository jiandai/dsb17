'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
ver 20170330 by jian: review output of pretained unet
ver 20170331 by jian: revamp using LUNA data, rename, and split the logic into two parts, pay more attention to numeric type & its conversion
ver 20170405 by jian: debug
ver 20170406 by jian: preproc test set
ver 20170408 by jian: preproc stage2 data

to-do:
'''

import numpy as np
import pandas as pd
from LUNA_segment_lung_ROI import segment_ROI,debugPlot
from dicom_batch import get_one_scan
#images_path = '../input/stage1/'
images_path = '../input/stage2/'
#output_path = '../process/tr-in/'
output_path = '../process/stage2/'

#labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id') # 1397 tr cases
#test_csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id') #198 cases
stage2_csv=pd.read_csv('../input/stage2_sample_submission.csv',index_col='id') #506 cases

import sys
# assume len(sys.argv)>1:
# assertion: input an integer


BATCH_INDEX = int(sys.argv[1])

#BATCH_SIZE = 35 # for training set /w 40 batches for 1397 records
#BATCH_SIZE = 25 # for test set /w 8 batches for 198 records
BATCH_SIZE = 17 # for test set /w 30 batches for 506 records

# test
#BATCH_INDEX = 10
#BATCH_SIZE =2

batch_start = (BATCH_INDEX-1)*BATCH_SIZE
batch_end = BATCH_INDEX*BATCH_SIZE



#batch_start=0 # saved
#batch_start=100
#batch_start=300
#batch_start=500 # saved
#batch_start=700 # saved
#batch_start=1000 # saved
#batch_start=570 # data issue /w 571-th

#batch_end=1
#batch_end=100
#batch_end=300
#batch_end=500
#batch_end=700
#batch_end=1000
#batch_end=2000

#patients = labels_csv.index[batch_start:batch_end]
#patients = test_csv.index[batch_start:batch_end]
patients = stage2_csv.index[batch_start:batch_end]
for i,pat in enumerate(patients):
	print i,pat
        scan,sp = get_one_scan(images_path+pat,resampling=False) #s1
        n_slice = scan.shape[0] #
	scan = scan.astype(np.float64)
	seged = []
	for j in range(n_slice)[:]: # 
		img,size = segment_ROI(scan[j],normalize=True,keep_size=True,to_square=False,resizing=False)
		if not img is None:
			seged.append(img)
	n_seg =len(seged)
	segs = np.ndarray([n_seg,1,512,512],dtype=np.float32)
	for j in range(n_seg):
		segs[j,0]= seged[j]
	np.save(output_path+pat+'-ROI-v3.npy', segs)

