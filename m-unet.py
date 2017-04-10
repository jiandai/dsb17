'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
ver 20170330 by jian: review output of pretained unet
ver 20170331 by jian: revamp using LUNA data,split the logic into two parts, pay more attention to numeric type & its conversion
ver 20170405 by jian: revamp
ver 20170407 by jian: save the extracted features, split the logic, set smooth=0
ver 20170408 by jian: batch for tr, tst, stage2
ver 20170409.1 by jian: new hdf5 file per exp2 to train unet
ver 20170409.2 by jian: design the parallel proc, 102 per batch, 2 for 198 range(1,3), 5 for 506 range(3,8), 14 for 1397 range(8,22)
to-do:
'''

import pandas as pd
import numpy as np


import sys
batch_number = int(sys.argv[1])
batch_size = 102
print 'batch_number=',batch_number
# batch number : 1-21
if batch_number in range(1,3):
	output_path = '../process/tr-in/'
	batch_start=(batch_number-1)*batch_size
	#batch_start=570 # data issue /w 571-th
	batch_end=(batch_number-0)*batch_size
	csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id')
	#feature_file = 'test_features.npy'
	feature_file = 'test_features-v2-b'+str(batch_number-2)+'.npy'

elif batch_number in range(3,8):
	output_path = '../process/stage2/'
	batch_start=(batch_number-3)*batch_size
	batch_end=(batch_number-2)*batch_size
	csv = pd.read_csv('../input/stage2_sample_submission.csv', index_col='id')
	#feature_file = 'stage2_features.npy'
	feature_file = 'stage2_features-v2-b'+str(batch_number-7)+'.npy'

elif batch_number in range(8,22):
	output_path = '../process/tr-in/'
	batch_start=(batch_number-8)*batch_size
	batch_end=(batch_number-7)*batch_size
	csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
	#feature_file = 'training_features.npy'
	feature_file = 'training_features-v2-b'+str(batch_number)+'.npy'


print batch_start,'--',batch_end
patients = csv.index[batch_start:batch_end]



from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512
#smooth = 1.
smooth = 0.

#h5file='unet-v5.hdf5'
h5file='unet-v4-861663.hdf5'
from LUNA_train_unet import dice_coef,dice_coef_np,dice_coef_loss,get_unet
model = get_unet()
model.load_weights('../DSB3Tutorial/tutorial_code/'+h5file) # hopefully the best shoot


from classify_nodes import getRegionMetricRow,getRegionFromMap

numfeatures = 9
feature_array = np.zeros((len(patients),numfeatures))
for i,pat in enumerate(patients[:]):
	print i,pat
	#scan = np.load(output_path+pat+'-ROI.npy')
	scan = np.load(output_path+pat+'-ROI-v3.npy')
        ### per scan normalization, vs per tr/tt normalization in unet fit
        mean = np.mean(scan)  # mean for data centering
        std = np.std(scan)  # std for data normalization
        scan -= mean  # images should already be standardized, but just in case
        scan /= std
        ###
        n_slice = scan.shape[0]
	segs=np.zeros([n_slice,1,512,512])
	for j in range(n_slice)[:]: # 
		predicted = model.predict(scan[j].reshape(1,1,512,512))
		segs[j] = predicted

	feature_array[i] = getRegionMetricRow(segs) #s4
print feature_array.shape
np.save(feature_file,feature_array)

print 'preprocessing and segmentation finished'
