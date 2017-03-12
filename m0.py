'''
ver 60170311 by jian: ref to https://www.kaggle.com/mumech/data-science-bowl-6017/loading-and-processing-the-sample-images 
    minimal CNN
'''
images_path = '../sample_images/'
import os
patients = os.listdir(images_path)

import pandas as pd
labels_csv = pd.read_csv('../stage1_labels.csv', index_col='id')
# ``pretended'' labels
train_labels = labels_csv.iloc[:len(patients),:].values

import numpy as np
import scipy.ndimage
import dicom

SIZE = 512
RESIZE = 128
Z_RESIZE=128
train_features = np.zeros([len(patients), 1, Z_RESIZE, RESIZE, RESIZE], np.float16)
for i,pat in enumerate(patients):
    slices = [dicom.read_file(images_path + pat +'/' +s) for s in os.listdir(images_path+pat)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    img = np.stack([s.pixel_array for s in slices])
    img[img == -2000] = 0
    img = scipy.ndimage.zoom(img.astype(np.float), RESIZE/SIZE)
    if img.shape[0]>=Z_RESIZE:
        img = img[:Z_RESIZE]
    else:
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

nn = keras.models.Sequential([
keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', input_shape=train_features.shape[1:], dim_ordering='th'),
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
nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=1)
'''
first pass:
C:\Anaconda3\lib\site-packages\scipy\ndimage\interpolation.py:568: UserWarning: From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed. "the returned array has changed.", UserWarning)
(20, 1, 128, 128, 128)
Using TensorFlow backend.
Train on 18 samples, validate on 2 samples
Epoch 1/1
18/18 [==============================] - 4869s - loss: 0.6869 - acc: 0.7222 - val_loss: 7.5440 - val_acc: 0.0000e+00
'''
