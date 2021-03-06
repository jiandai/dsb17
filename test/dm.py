# version < 20170301 by jian
# version 20170301 by jian: run on server
# version 20170305 by jian: metadata of stage 1 data
# version 20170309 by jian: deal /w pandas new column, install pydicom on server for user
# version 20170310 by jian: test on iterating through `all' scans
# version 20170313 by jian: data check (orientation, spacing, etc)







import os
import dicom
image_path = '../input/sample_images/'
#image_path = '../input/stage1/'
for pat in os.listdir(image_path):
	scan = [dicom.read_file(image_path+pat+'/'+f) for f in os.listdir(image_path+pat)]
	S=len(scan)
	Thickness =  (scan[0].ImagePositionPatient[2] - scan[S-1].ImagePositionPatient[2])/(S-1)
	print pat,Thickness
	
quit()








# case check per Liz
#case ='../tmp/af2b9f7d489ffab8e3590658853cacf8/'
case ='../input/sample_images/0a099f2549429d29b32f349e95fb2244/'
case_slices=[dicom.read_file(case+f) for f in os.listdir(case)]
case_slices.sort(key=lambda x:x.InstanceNumber)
prev_pos,prev_loc=0,0
for j,slice in enumerate(case_slices):
	if j>0:
		pos_diff=slice.ImagePositionPatient[2]-prev_pos
		loc_diff=slice.SliceLocation-prev_loc
		print slice.InstanceNumber,slice.ImagePositionPatient[2],slice.SliceLocation,pos_diff,loc_diff
	prev_pos=slice.ImagePositionPatient[2]
	prev_loc=slice.SliceLocation
'''
2 -1339.80005 340.70 -1.30005 -1.3
3 -1341.09998 339.40 -1.29993 -1.3
4 -1342.40002 338.10 -1.30004 -1.3
5 -1343.69995 336.80 -1.29993 -1.3
6 -1345 335.50 -1.30005 -1.3
7 -1346.30005 334.20 -1.30005 -1.3
8 -1347.59998 332.90 -1.29993 -1.3
9 -1348.90002 331.60 -1.30004 -1.3
10 -1350.19995 330.30 -1.29993 -1.3
'''
# lesson is ImagePositionPatient[2] is preferred, and SliceLocation may be less accurate
quit()

# all data
#folders = os.listdir('../data/stage1/') # local
folders = os.listdir('../input/stage1/') # server path
print folders[0]
quit()



import sys
# assume len(sys.argv)>1:
# assertion: input an integer
BATCH_INDEX = int(sys.argv[1]) 
BATCH_SIZE = 3
batch_range = range((BATCH_INDEX-1)*BATCH_SIZE,BATCH_INDEX*BATCH_SIZE)
npz_path = '../process/v'+sys.argv[1]+'.npz'
import numpy
# use numpy.load to load
loaded=numpy.load(npz_path) # of type <class 'numpy.lib.npyio.NpzFile'>

#print loaded # <numpy.lib.npyio.NpzFile object at 0x2aad88fd1a90>
#print loaded.keys() # ['arr_0']
#print(loaded['arr_0'].shape) # (3,)
#print(type(loaded['arr_0'][0])) #<type 'numpy.ndarray'>
# loaded by itself cannot be iterated
for arr in loaded['arr_0']:
	print arr.shape
#print(loaded['arr_0'][0].shape)
#print(loaded['arr_0'][1].shape)
#print(loaded['arr_0'][2].shape)

quit()


from dicom_batch import get_one_scan
vox_list=[]
for i in batch_range:
	id = folders[i]
	print 'processing '+id+' ...'
	voxels = get_one_scan('../input/stage1/'+ id)
	vox_list.append(voxels)

print len(vox_list)
numpy.savez_compressed(npz_path,vox_list)
# 153444485b for 3 scans
quit()
'''
simple batch
1-f: 7.61 sec
2-f: 8.23 sec
3-f: 9.53 sec
4-f: 9.75 sec
5-f: 10.14 sec

batch array config 1: 5 scans * 4
log/preproc.o581641:40:    CPU time :               10.62 sec.
log/preproc.o581642:40:    CPU time :               33.62 sec.
log/preproc.o581643:40:    CPU time :               34.80 sec.
log/preproc.o581644:40:    CPU time :               48.70 sec.

batch array config 2: 5 scans * 6
log/preproc-arr-o-58176-1:39:    CPU time :               10.18 sec.
log/preproc-arr-o-58176-2:39:    CPU time :               10.52 sec.
log/preproc-arr-o-58176-3:39:    CPU time :               11.16 sec.
log/preproc-arr-o-58176-4:39:    CPU time :               10.65 sec.
log/preproc-arr-o-58176-5:39:    CPU time :               34.51 sec.
log/preproc-arr-o-58176-6:39:    CPU time :               33.84 sec.
'''






import pandas as pd
# training labels
#stg1_labels = pd.read_csv('../data/stage1_labels.csv')
stg1_labels = pd.read_csv('../input/stage1_labels.csv') # server path



# data check
print(set(stg1_labels.id).difference(folders)) # set()
# test set
print(len(set(folders).difference(stg1_labels.id)))

# check the count
print(stg1_labels.shape)
print(len(folders))

'''
there are totally 1595 cases
1397 training cases, and
198 test cases
'''

# training case labels:
print(stg1_labels.groupby('cancer').size()) # or use "stg1_labels.cancer.value_counts()"
'''
cancer
0    1035
1     362
'''
fileDF = pd.DataFrame(folders,columns=['id'])

print(stg1_labels.columns)

scanDF = fileDF.set_index('id').join(stg1_labels.set_index('id'),how='left')
print(scanDF.groupby('cancer').size())


scanDF['n_x'] = -1
scanDF['n_y'] = -1
scanDF['n_z'] = -1
scanDF['d_x'] = -1
scanDF['d_y'] = -1
scanDF['d_z'] = -1
scanDF['intercept'] = -1
scanDF['slope'] = -1


testDF = scanDF[pd.isnull(scanDF.cancer)].copy()
trainingDF = scanDF[pd.notnull(scanDF.cancer)].copy() # should be the "same" as stg1_labels
# ref the use of ".copy()" to http://stackoverflow.com/questions/37435468/pandas-settingwithcopywarning-when-using-a-subset-of-columns?rq=1






# Iterate through training set
all_scan=[]
for id in trainingDF.index[:1]:
	#slices = get_one_scan('../data/stage1/'+id)
	slices = get_one_scan('../input/stage1/'+id)

	trainingDF.loc[id,'n_x'] = slices[0].Rows
	trainingDF.loc[id,'n_y'] = slices[0].Columns
	trainingDF.loc[id,'n_z'] = len(slices)
	trainingDF.loc[id,'d_x'] = slices[0].PixelSpacing[0]
	trainingDF.loc[id,'d_y'] = slices[0].PixelSpacing[1]
	trainingDF.loc[id,'intercept'] = slices[0].RescaleIntercept
	trainingDF.loc[id,'slope'] = slices[0].RescaleSlope
	for s,slice in enumerate(slices):
		print([slice.InstanceNumber]+slice.ImagePositionPatient + [slice.SliceLocation,str(slice.Rows),str(slice.Columns)]+slice.PixelSpacing+[slice.RescaleIntercept,slice.RescaleSlope])
	all_scan.append(slices)
	# To apply pydicom to read in the slice folders
print(trainingDF.head())
#trainingDF.n_slices.max()
#541

#trainingDF.n_slices.min()
#94
