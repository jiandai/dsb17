'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
ver 20170330 by jian: review output of pretained unet
ver 20170331 by jian: revamp, numeric type is very important, save seg feature dataset

to-do:

'''


import numpy as np
import pandas as pd
from LUNA_segment_lung_ROI import segment_ROI,debugPlot
from dicom_batch import get_one_scan
images_path = '../input/stage1/'
output_path = '../process/tr-in/'
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
# 1397 tr cases
batch_start=0 # saved
batch_end=2000



#batch_start=100
#batch_start=300
#batch_start=500 # saved
#batch_start=700 # saved
#batch_start=1000 # saved
#batch_start=570 # data issue /w 571-th

#batch_end=100
#batch_end=300
#batch_end=500
#batch_end=700
#batch_end=1000
#batch_end=2000


patients = labels_csv.index[batch_start:batch_end]



#from keras.models import Model
#from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
#from keras.optimizers import Adam
#from keras.optimizers import SGD
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as K
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code



img_rows = 512
img_cols = 512
smooth = 1.

from LUNA_train_unet import dice_coef,dice_coef_np,dice_coef_loss,get_unet



model = get_unet()
model.load_weights('../DSB3Tutorial/unet.hdf5') # pretrained unet


from classify_nodes import getRegionMetricRow,getRegionFromMap,logloss,classifyData
numfeatures = 9
feature_array = np.zeros((len(patients),numfeatures))
for i,pat in enumerate(patients):
	print i,pat
        scan,sp = get_one_scan(images_path+pat,resampling=False) #s1
        n_slice = scan.shape[0]
	#print scan.dtype # int16
	scan = scan.astype(np.float64)

	segs=np.zeros([n_slice,1,512,512])
	for j in range(n_slice)[:]: # 
		img,size = segment_ROI(scan[j],normalize=True) # return float64
		if not img is None:
			img = img.reshape(1,1,512,512)
			img = img.astype(np.float32)
			segs[j] = model.predict(img) [0] # s3, please review this part
			#segs[j] = img

	np.save(output_path+pat+'.npy', segs)
	feature_array[i] = getRegionMetricRow(segs) #s4

print 'preprocessing and segmentation finished -- feature array shape:',feature_array.shape
np.save('seg-out-feature-array.npy',feature_array)
truth_metric = labels_csv.cancer[batch_start:batch_end]
classifyData(feature_array,truth_metric) #s5
