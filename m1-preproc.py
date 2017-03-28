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
ver 60170320 by jian: allow diff pixel spacing when resampling
ver 60170327.1 by jian: add segmentation, read-seg-chop-resample
ver 60170327.2 by jian: folk from m0-preproc.py

'''
import sys
# assume len(sys.argv)>1:
# assertion: input an integer
BATCH_INDEX = int(sys.argv[1])
#BATCH_INDEX = 10
#BATCH_SIZE =25 # for test set /w 8 batches for 198 records
BATCH_SIZE = 35 # for training set /w 40 batches for 1397 records
#BATCH_SIZE =2 
batch_range = range((BATCH_INDEX-1)*BATCH_SIZE,BATCH_INDEX*BATCH_SIZE)
batch_start = (BATCH_INDEX-1)*BATCH_SIZE
batch_end = BATCH_INDEX*BATCH_SIZE

PIXEL_SPACING=1
#PIXEL_SPACING=2.5
#PIXEL_SPACING=1.5

#N=1500 # more than the number of cases

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
#test_csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id')


import os
import numpy as np
import scipy.ndimage
import dicom
import matplotlib.pyplot as plt

patients = labels_csv.index[batch_start:batch_end][:]
#patients = test_csv.index[batch_start:batch_end]

print patients
print batch_start,batch_end
print batch_range
print len(patients)

from dicom_batch import get_one_scan,resampling_one
from dicom_batch import segment_lung_mask
from LUNA_segment_lung_ROI import segment_ROI
from skimage import measure
#vox_list=[]
#log=[]
images_path = '../input/stage1/'
#output_path = '../process/prep-out/test/'
output_path = '../process/prep-out/training/'
#out_file_note = '-simple' # for test
out_file_note = '-r-s-r' # for read-seg-resample
#out_file_note = '-3d-seg' # for read-seg-resample
for i,pat in enumerate(patients):
	print i,pat
	#img = get_one_scan(images_path+pat,resampling=True,new_spacing=[PIXEL_SPACING,PIXEL_SPACING,PIXEL_SPACING])
	img,spacing = get_one_scan(images_path+pat,resampling=False)
	print 'original size and spacing:',img.shape, spacing
	
	# user LUNA_segment_lung_ROI as 2d-segmenter: input img, output segs
	seg_list=[]
	min_row=[]
	max_row=[]
	min_col=[]
	max_col=[]
	for j in range(img.shape[0])[:]:
		seg,box = segment_ROI(img[j],keep_size=True,to_square=False,resizing=False)
		if seg is not None:
			seg_list.append(seg)
			min_row.append(box[0])
			max_row.append(box[1])
			min_col.append(box[2])
			max_col.append(box[3])
	segs = np.stack(seg_list)
	overall_min_row = min(min_row)
	overall_max_row = max(max_row)
	overall_min_col = min(min_col)
	overall_max_col = max(max_col)
	print 'output size:',segs.shape
	segs = segs[:,overall_min_row:overall_max_row,overall_min_col:overall_max_col]
	print 'chopped size:',segs.shape

	# user segment_lung_mask
	'''
	mask = segment_lung_mask(img)
	print 'mask size:',mask.shape
	labels = measure.label(mask)
	regions = measure.regionprops(labels)
	min_depth=img.shape[0]
	max_depth=0
	min_row = img.shape[1]
	max_row = 0
	min_col = img.shape[2]
	max_col = 0
	for prop in regions:
	        B = prop.bbox
	        if min_depth > B[0]:
	            min_depth = B[0]
	        if min_row > B[1]:
	            min_row = B[1]
	        if min_col > B[2]:
	            min_col = B[2]
	        if max_depth < B[3]:
	            max_depth = B[3]
	        if max_row < B[4]:
	            max_row = B[4]
	        if max_col < B[5]:
	            max_col = B[5]
	
	print(min_depth, max_depth, min_row, max_row, min_col, max_col)
	print(-min_depth+max_depth, -min_row+max_row, -min_col+max_col)
	segs = (img * mask) [min_depth:max_depth, min_row:max_row, min_col:max_col]
	print 'chopped size:',segs.shape
	'''

	segs = resampling_one(segs,spacing,new_spacing=[PIXEL_SPACING,PIXEL_SPACING,PIXEL_SPACING])
	print 'resampled size:',segs.shape
	#plt.hist(segs.flatten(),bins=100);plt.show()

	#log.append([i,pat,img.shape[0],img.shape[1],img.shape[2]])
	#vox_list.append(img)
	# indiviudal scan saving
	npy_path = output_path+pat+out_file_note+'.npy'
	np.save(npy_path,segs)
	# indiviudal scan saving
	#npz_path = output_path+pat+'.npz'
	#np.savez_compressed(npz_path,img)




#logDF=pd.DataFrame(log,columns=['index','id','sz','sx','sy'])
#print logDF
#logDF.to_csv("preproc-test-set-batch-"+sys.argv[1]+"-log.csv")
#logDF.to_csv("preproc-training-set-batch-"+sys.argv[1]+"-log.csv")
#logDF.to_csv("preproc-log.csv")
#npz_file = "preproc-test-set-batch-"+sys.argv[1]+".npz"
#npz_file = "preproc-training-set-res-"+str(PIXEL_SPACING)+"-batch-"+sys.argv[1]+".npz"
#print npz_file
#npz_path = output_path + npz_file
#np.savez_compressed(npz_path,vox_list)


