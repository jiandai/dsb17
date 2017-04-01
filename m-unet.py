'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
ver 20170330 by jian: review output of pretained unet
ver 20170331 by jian: revamp using LUNA data,split the logic into two parts, pay more attention to numeric type & its conversion

to-do:
'''

import pandas as pd
import numpy as np
output_path = '../process/tr-in/'
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')

batch_start=0
#batch_start=570 # data issue /w 571-th
batch_end=200



patients = labels_csv.index[batch_start:batch_end]
truth_metric = labels_csv.cancer[batch_start:batch_end]






from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


K.set_image_dim_ordering('th')  # Theano dimension ordering in this code



img_rows = 512
img_cols = 512
smooth = 1.

from LUNA_train_unet import dice_coef,dice_coef_np,dice_coef_loss,get_unet



model = get_unet()
model.load_weights('../DSB3Tutorial/tutorial_code/unet.hdf5')




from classify_nodes import getRegionMetricRow,getRegionFromMap,logloss,classifyData
numfeatures = 9
feature_array = np.zeros((len(patients),numfeatures))
for i,pat in enumerate(patients[:]):
	print i,pat
	scan = np.load(output_path+pat+'-ROI.npy')
        n_slice = scan.shape[0]
	segs=np.zeros([n_slice,1,512,512])
	for j in range(n_slice)[:]: # 
		predicted = model.predict(scan[j].reshape(1,1,512,512))
		segs[j] = predicted

	feature_array[i] = getRegionMetricRow(segs) #s4

print 'preprocessing and segmentation finished'
classifyData(feature_array,truth_metric) #s5
