'''

Naive or even ``non-sense'' CNN

ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
ver 60170312 by jian: run on server, 128*512*512 kills tf on server
ver 60170313.1 by jian: 1st pass of all cases
ver 60170313.2 by jian: consolidate preproc, check the xy-size, orientation
ver 60170313.3 by jian: resample and save to npz
ver 60170313.4 by jian: split the script
ver 60170314 by jian: test to read mini batch npz files
ver 60170315 by jian: compare mini batch loading vs single loading, compare mini back loading training vs single loading training
ver 60170316 by jian: maximize the training shape by pushing the limit of gpu mem
ver 60170317 by jian: encounter mem bottleneck due to sequential objects, refator the prep to shrink mem footprint
ver 60170318 by jian: the training of large cnn takes much longer, review CNN math base
ver 60170320 by jian: use 2.5 mm resolution for a test
ver 60170322 by jian: add normalization
ver 60170323 by jian: add prediction

to-do: 
=> add dropout layer
=> multi-gpu training via tf
*=> use joblib of py
=> add CV / model selection
=> save trained model


'''

import pandas as pd

#import glob
#sz_min,sz_max,sx_min,sx_max,sy_min,sy_max = 10000,0,10000,0,10000,0
#for f in glob.glob('preproc-training-set-batch*'):
#	tr_meta = pd.read_csv(f)
#	sz_min = min(sz_min,tr_meta.sz.min())
#	sz_max = max(sz_max,tr_meta.sz.max())
#	sx_min = min(sx_min,tr_meta.sx.min())
#	sx_max = max(sx_max,tr_meta.sx.max())
#	sy_min = min(sy_min,tr_meta.sy.min())
#	sy_max = max(sy_max,tr_meta.sy.max())
#print sz_min,sz_max
#print sx_min,sx_max
#print sy_min,sy_max
'''
57 171
100 196
100 196
'''

import numpy as np


#tr_meta = pd.read_csv('../process/prep-out/preproc-log.csv')
#print tr_meta.sz.min(),tr_meta.sz.max()
#print tr_meta.sx.min(),tr_meta.sx.max()
#print tr_meta.sy.min(),tr_meta.sy.max()

'''
min, max range of resampled scans:
142 428 for z
250 490 for x
250 490 for y
'''







######################################################################################

# Single batch load: 
'''
test job 99793
Started at Wed Mar 15 12:46:35 2017
Results reported on Wed Mar 15 13:04:30 2017
~18 min
780.78 sec cpu time  < physical time
/w m0.b configuration
'''
#npz_path = '../process/prep-out/resampled-out.npz'
#batch=np.load(npz_path) # of type <class 'numpy.lib.npyio.NpzFile'>
#pre_processed=list(batch['arr_0'])
#N=1500
#N=20

######################################################################################


# Mulpi-batch load:
'''
test job 100002
Started at Wed Mar 15 13:09:41 2017
Results reported on Wed Mar 15 13:26:48 2017
~17 min
773.01 sec cpu time 
/w m0.b
'''


BATCH_SIZE=35 # for training batch
S=2 # number of batches to be used, max=40
#S=40 # number of batches to be used, max=40
N=BATCH_SIZE * S 

'''
128*490*490 OOM
128*280*280 OOM
256*256*256 OOM
200*256*256 OOM
160*256*256 OOM

206304: 128*300*300 OOM
287050,289839: 128*300*300 ok /w K80
290200: 128*320^2 ok /w K80
289655,289816: 128*360*360 OOM
290606: 160*320*320 OOM
290746: 160*300*300 ok
'''
#Z_RESIZE=160
Z_RESIZE=171
#N_XY=300
N_XY=196
X_RESIZE = N_XY
Y_RESIZE = N_XY

from dicom_batch import normalize
#pre_processed = []


#tr_prefix = '../process/prep-out/training/preproc-training-set-batch-'
tr_prefix = '../process/prep-out/training/preproc-training-set-res-2.5-batch-'
for j in range(1,41)[:S]:
	npz_path = tr_prefix+str(j)+'.npz'
	#batch=np.load(npz_path) # of type <class 'numpy.lib.npyio.NpzFile'>
	#pre_processed=pre_processed + list(batch['arr_0'])
	pre_processed=list(np.load(npz_path) ['arr_0'])
	train_features_b = np.zeros([len(pre_processed), 1, Z_RESIZE, X_RESIZE, Y_RESIZE], np.float16)


	for i,img in enumerate(pre_processed):
	        print i,img.shape
	
		if img.shape[0]>=Z_RESIZE:
			# cut extra
			img = img[(img.shape[0]-Z_RESIZE)//2:(img.shape[0]-Z_RESIZE)//2 + Z_RESIZE]
		else:
			# padding on above and below
			img = np.concatenate([
				np.zeros([(Z_RESIZE- img.shape[0])//2, img.shape[1], img.shape[2]], np.float16),
				img, 
				np.zeros([Z_RESIZE- img.shape[0]-(Z_RESIZE- img.shape[0])//2, img.shape[1], img.shape[2]], np.float16) ],axis=0)
	        #print '-',img.shape
		if img.shape[1]>=X_RESIZE:
			# cut extra
			img = img[:,(img.shape[1]-X_RESIZE)//2:(img.shape[1]-X_RESIZE)//2 + X_RESIZE,:]
		else:
			# padding on above and below
			pad0 = np.zeros([Z_RESIZE,(X_RESIZE- img.shape[1])//2, img.shape[2]], np.float16)
			pad1 = np.zeros([Z_RESIZE,X_RESIZE- img.shape[1]-(X_RESIZE- img.shape[1])//2, img.shape[2]], np.float16)
			img = np.concatenate([
				pad0,
				img, 
				pad1],axis=1)
	        #print '-',img.shape
		if img.shape[2]>=Y_RESIZE:
			# cut extra
			img = img[:,:,(img.shape[2]-Y_RESIZE)//2:(img.shape[2]-Y_RESIZE)//2 + Y_RESIZE]
		else:
			# padding on above and below
			img = np.concatenate([
				np.zeros([Z_RESIZE,X_RESIZE,(Y_RESIZE- img.shape[2])//2], np.float16),
				img, 
				np.zeros([Z_RESIZE,X_RESIZE,Y_RESIZE- img.shape[2]-(Y_RESIZE- img.shape[2])//2], np.float16) ],axis=2)
	
	        #print '-',img.shape
		train_features_b[i] = normalize(img).reshape([1, Z_RESIZE, X_RESIZE, Y_RESIZE])



	print (train_features_b.shape)
	if j==1:
		train_features = train_features_b
	else:
		train_features = np.concatenate([train_features,train_features_b])

print (train_features.shape)








######################################################################################


import keras
# `` non-sense '' CNN
input_shape=train_features.shape[1:]
nn = keras.models.Sequential([
keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', input_shape=input_shape, dim_ordering='th'),
keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
keras.layers.convolutional.Convolution3D(128, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),
keras.layers.convolutional.Convolution3D(256, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),
keras.layers.convolutional.AveragePooling3D((4, 4, 4), dim_ordering='th'),
keras.layers.core.Flatten(),
keras.layers.core.Dense(32, activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.core.Dense(1, activation='sigmoid')
])
nn.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
print(nn.summary())



######################################################################################

#N_EPOCH=10
N_EPOCH=1

labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
train_labels = labels_csv.iloc[:N,:].values
nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=N_EPOCH)
# prediction
predict = nn.predict(train_features)
print type(predict)
print predict

