import os
# all data
files = os.listdir('../data/stage1/')
import pandas as pd
# training labels
stg1_labels = pd.read_csv('../data/stage1_labels.csv')

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
print(stg1_labels.groupby('cancer').size())
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
for id in trainingDF.index:
	slices = os.listdir('../data/stage1/'+id)
	print(id+'\t'+str(len(slices)))
	# To apply pydicom to read in the slice files
