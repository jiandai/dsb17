# version < 20170301 by jian
# version 20170301 by jian: run on server
# version 20170305 by jian: metadata of stage 1 data
import os
# all data
#files = os.listdir('../data/stage1/') # local
files = os.listdir('../download/stage1/') # server path

import pandas as pd
# training labels
#stg1_labels = pd.read_csv('../data/stage1_labels.csv')
stg1_labels = pd.read_csv('../download/stage1_labels.csv') # server path

# data check
print(set(stg1_labels.id).difference(files)) # set()
# test set
print(len(set(files).difference(stg1_labels.id)))

# check the count
print(stg1_labels.shape)
print(len(files))

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
fileDF = pd.DataFrame(files,columns=['id'])
print(fileDF.columns)
print(stg1_labels.columns)
scanDF = fileDF.set_index('id').join(stg1_labels.set_index('id'),how='left')
print(scanDF.groupby('cancer').size())
testDF = scanDF[pd.isnull(scanDF.cancer)]
trainingDF = scanDF[pd.notnull(scanDF.cancer)] # should be the "same" as stg1_labels

# Iterate through training set
print(trainingDF)
trainingDF['n_slices']=0
from dicom_batch import get_one_scan
all_scan=[]
for id in trainingDF.index[:]:
	#slices = get_one_scan('../data/stage1/'+id)
	slices = get_one_scan('../download/stage1/'+id)
	trainingDF.loc[id,'n_slices'] = len(slices)
	#all_scan.append(slices)
	# To apply pydicom to read in the slice files
print(trainingDF)
#trainingDF.n_slices.max()
#541

#trainingDF.n_slices.min()
#94
