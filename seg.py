'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
'''


import numpy as np

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




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = get_unet()
model.load_weights('../DSB3Tutorial/unet.hdf5')

from LUNA_segment_lung_ROI import segment_ROI,debugPlot
from dicom_batch import get_one_scan

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
batch_start=0
#batch_start=570 # data issue /w 571-th
#batch_end=2000
batch_end=100
patients = labels_csv.index[batch_start:batch_end]
truth_metric = labels_csv.cancer[batch_start:batch_end]


from classify_nodes import getRegionMetricRow,getRegionFromMap,logloss,classifyData
images_path = '../input/stage1/'
numfeatures = 9
feature_array = np.zeros((len(patients),numfeatures))
for i,pat in enumerate(patients):
	print i,pat
        scan = get_one_scan(images_path+pat,resampling=False)
        n_slice = scan.shape[0]
	segs=np.zeros([n_slice,1,512,512])
	for j in range(n_slice)[:]: # 
		img = segment_ROI(scan[j])
                # visual test here
                #debugPlot(scan[j])
                #debugPlot(img)
		if not img is None:
			img = img.reshape(1,1,512,512).astype(np.float32)
			segs[j] = model.predict(img) [0] # please review this part
        print(segs.shape)
	feature_array[i] = getRegionMetricRow(segs)

print 'preprocessing and segmentation finished'
classifyData(feature_array,truth_metric)
