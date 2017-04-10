
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

working_path = '../../../../luna16/processed/'
img_rows = 512
img_cols = 512


def dice_coef(y_true, y_pred,smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred,smooth=0.):
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




#imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
#imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)

#imgs_train = np.load(working_path+"trainImages-v2.npy").astype(np.float32)
#imgs_mask_train = np.load(working_path+"trainMasks-v2.npy").astype(np.float32)
imgs_train = np.load(working_path+"trainImages-v3.npy").astype(np.float32)
imgs_mask_train = np.load(working_path+"trainMasks-v3.npy").astype(np.float32)

#print imgs_train.shape,imgs_train.dtype
#(2782, 1, 512, 512) float32
#print imgs_mask_train.shape,imgs_mask_train.dtype
#(2782, 1, 512, 512) float32
#print imgs_train.min(),imgs_train.max()
#-0.567004 0.934722
#print np.histogram(imgs_mask_train)
#(array([725854680,     54780,     49358,     46171,     42965,     42088,
#           44223,     47299,     52203,   3050841]), array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]))



model0 = get_unet();model0.load_weights('../DSB3Tutorial/unet.hdf5')
model1 = get_unet();model1.load_weights('../DSB3Tutorial/tutorial_code/unet.hdf5')
model2 = get_unet();model2.load_weights('../DSB3Tutorial/tutorial_code/unet-v2.hdf5')
model5 = get_unet();model5.load_weights('../DSB3Tutorial/tutorial_code/unet-v5.hdf5')






for case in range(10):
	y_pred0 = model0.predict(imgs_train[case].reshape(1,1,512,512), verbose=0)
	y_pred1 = model1.predict(imgs_train[case].reshape(1,1,512,512), verbose=0)
	y_pred2 = model2.predict(imgs_train[case].reshape(1,1,512,512), verbose=0)
	y_pred5 = model5.predict(imgs_train[case].reshape(1,1,512,512), verbose=0)
	#print y_pred.shape #(1, 1, 512, 512)
	print case
	print dice_coef_np(imgs_mask_train[case,0],y_pred0[0,0]), ':',y_pred0.min(),y_pred0.max()
	print dice_coef_np(imgs_mask_train[case,0],y_pred1[0,0]), ':',y_pred1.min(),y_pred1.max()
	print dice_coef_np(imgs_mask_train[case,0],y_pred2[0,0]), ':',y_pred2.min(),y_pred2.max()
	print dice_coef_np(imgs_mask_train[case,0],y_pred5[0,0]), ':',y_pred5.min(),y_pred5.max()
	print np.histogram(y_pred0)
	print np.histogram(y_pred1)
	print np.histogram(y_pred2)
	print np.histogram(y_pred5)



quit()
# ver 20170330 by jian: test script to compare original scan and output from pretrained unet
import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')

patients = labels_csv.index
images_path = '../input/stage1/'
output_path = '../process/tr-in/'

from dicom_batch import get_one_scan
import numpy as np

def get_pair(j):
	pat = patients[j]
        scan,sp = get_one_scan(images_path+pat,resampling=False) #s1
	segs = np.load(output_path+pat+'.npy')
	segs = segs.reshape(segs.shape[0],segs.shape[2],segs.shape[3])
	print scan.shape
	print segs.shape
	return scan,segs


import matplotlib.pyplot as plt
def tstPlot(a,b):
	fig,ax = plt.subplots(1,2,figsize=[5,5])
	ax[0].imshow(a,cmap='gray')
	ax[1].imshow(b,cmap='gray')
	plt.show()
def tstPlot(a,b,n):
	fig,ax = plt.subplots(1,2,figsize=[5,5])
	ax[0].imshow(a[n],cmap='gray')
	ax[1].imshow(b[n],cmap='gray')
	plt.show()

scan,segs = get_pair(9)
tstPlot(scan,segs,44)


