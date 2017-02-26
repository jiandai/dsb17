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
quit()
print(pd.DataFrame(files,columns=['id']).columns)
print(stg1_labels.head())
