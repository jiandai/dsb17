'''
ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
ver 60170312 by jian: run on server, 128*512*512 kills tf
    minimal CNN
'''

N=1500 # more than the number of cases
N_EPOCH=10

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
patients = labels_csv.iloc[:N,:].index
train_labels = labels_csv.iloc[:N,:].values
images_path = '../input/stage1/'



import os
import numpy as np
import scipy.ndimage
import dicom

SIZE = 512
RESIZE = 128
resize_factor = RESIZE*1.0/SIZE

Z_RESIZE=128
train_features = np.zeros([len(patients), 1, Z_RESIZE, RESIZE, RESIZE], np.float16)
for i,pat in enumerate(patients):
    slices = [dicom.read_file(images_path + pat +'/' +s) for s in os.listdir(images_path+pat)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    img = np.stack([s.pixel_array for s in slices])
    img[img == -2000] = 0

    # resizing / resampling
    #img = scipy.ndimage.zoom(img.astype(np.float), RESIZE/SIZE)
    img = scipy.ndimage.interpolation.zoom(img.astype(np.float), resize_factor)

    if img.shape[0]>=Z_RESIZE:
        img = img[:Z_RESIZE]
    else:
	# padding
        img = np.concatenate([
            np.zeros([(Z_RESIZE- img.shape[0])//2, RESIZE, RESIZE], np.float16),
            img, 
            np.zeros([Z_RESIZE- img.shape[0]-(Z_RESIZE- img.shape[0])//2, RESIZE, RESIZE], np.float16)
            ])
    img = img.reshape(1,Z_RESIZE,RESIZE,RESIZE)
    train_features[i] = img
  
print (train_features.shape)

# `` non-sense '' CNN
import keras
input_shape=train_features.shape[1:]
#input_shape=(1,128,512,512)
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

nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=N_EPOCH)
'''
first pass on local:
C:\Anaconda3\lib\site-packages\scipy\ndimage\interpolation.py:568: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed. "the returned array has changed.", UserWarning)
(20, 1, 128, 128, 128)
Using TensorFlow backend.
Train on 18 samples, validate on 2 samples
Epoch 1/1
18/18 [==============================] - 4869s - loss: 0.6869 - acc: 0.7222 - val_loss: 7.5440 - val_acc: 0.0000e+00
'''
