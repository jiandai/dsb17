'''
Naive or even ``non-sense'' CNN

ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
ver 60170312 by jian: run on server, 128*512*512 kills tf on server
ver 60170313.1 by jian: 1st pass of all cases
ver 60170313.2 by jian: consolidate preproc, check the xy-size, orientation
ver 60170313.3 by jian: resample and save to npz
ver 60170313.4 by jian: split the script

to-do: 
=> use multiple batches of pre-proced data
=> use a test training shape to have a pass
=> optimize the training shape
=> add prediction

'''


import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')

#tr_meta = pd.read_csv('preproc-log.csv')
#print tr_meta.sz.min(),tr_meta.sz.max()
#print tr_meta.sx.min(),tr_meta.sx.max()
#print tr_meta.sy.min(),tr_meta.sy.max()
'''
142 428 for z
250 490 for x
250 490 for y
'''

import numpy as np
npz_path = '../process/prep-out/resampled-out.npz'
pre_processed=np.load(npz_path) # of type <class 'numpy.lib.npyio.NpzFile'>

#print pre_processed # <numpy.lib.npyio.NpzFile object at 0x2aad88fd1a90>
#print pre_processed.keys() # ['arr_0']
#print(pre_processed['arr_0'].shape) # (3,)
#print(type(pre_processed['arr_0'][0])) #<type 'numpy.ndarray'>

# pre_processed by itself cannot be iterated


#N_EPOCH=10
N_EPOCH=1
#N=20
N=1500

Z_RESIZE=128
X_RESIZE = 128
Y_RESIZE = 128
train_features = np.zeros([len(pre_processed['arr_0']), 1, Z_RESIZE, X_RESIZE, Y_RESIZE], np.float16)
for i,img in enumerate(pre_processed['arr_0'][:N]):
        print i,img.shape

	'''
	if img.shape[0]>=Z_RESIZE:
		# cut extra
		img = img[(img.shape[0]-Z_RESIZE)//2:(img.shape[0]-Z_RESIZE)//2 + Z_RESIZE]
	else:
		# padding on above and below
		img = np.concatenate([
			np.zeros([(Z_RESIZE- img.shape[0])//2, RESIZE, RESIZE], np.float16),
			img, 
			np.zeros([Z_RESIZE- img.shape[0]-(Z_RESIZE- img.shape[0])//2, RESIZE, RESIZE], np.float16) ])
		img = img.reshape(1,Z_RESIZE,RESIZE,RESIZE)
	'''
	# in this test, no need to pad 
	img = img[(img.shape[0]-Z_RESIZE)//2:(img.shape[0]-Z_RESIZE)//2 + Z_RESIZE]
	img = img[:,(img.shape[0]-X_RESIZE)//2:(img.shape[0]-X_RESIZE)//2 + X_RESIZE,:]
	img = img[:,:,(img.shape[0]-Y_RESIZE)//2:(img.shape[0]-Y_RESIZE)//2 + Y_RESIZE]

	train_features[i] = img.reshape([1, Z_RESIZE, X_RESIZE, Y_RESIZE])



print (train_features.shape)




# `` non-sense '' CNN
import keras
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





train_labels = labels_csv.iloc[:N,:].values
nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=N_EPOCH)
# => add prediction protion here

'''
first pass on local:
C:\Anaconda3\lib\site-packages\scipy\ndimage\interpolation.py:568: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed. "the returned array has changed.", UserWarning)
(20, 1, 128, 128, 128)
Using TensorFlow backend.
Train on 18 samples, validate on 2 samples
Epoch 1/1
18/18 [==============================] - 4869s - loss: 0.6869 - acc: 0.7222 - val_loss: 7.5440 - val_acc: 0.0000e+00
'''
