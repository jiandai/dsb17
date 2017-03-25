import numpy as np
#img_file = "test-img.npy"
#seg_file = "test-seg.npy"
#seg_file = '../../DSB3Tutorial/tutorial_code/masksTestPredicted.npy'
#img_file = '../../DSB3Tutorial/tutorial_code/pre-processed-images.npy'
seg_file = '../../DSB3Tutorial/tutorial_code/masksTestPredicted-test-2.npy'
img_file = '../../DSB3Tutorial/tutorial_code/pre-processed-images-test-2.npy'
imgs = np.load(img_file)
out = np.load(seg_file)

import matplotlib.pyplot as plt
#n=imgs.shape[0]
n=3
fig,ax = plt.subplots(n,3,figsize=[8,8])
for j in range(n)[:]:
	ax[j,0].imshow(imgs[j,0,:,:],cmap='gray')
	ax[j,1].imshow(out[j,0,:,:],cmap='gray')
	ax[j,2].imshow(imgs[j,0,:,:]*out[j,0,:,:],cmap='gray') # this multiplication may not make any sense
plt.show()



# Test code for different purpose
#import os
#print os.environ['DISPLAY']

#data_path = '../input/sample_images/'
#patients = os.listdir(data_path)
#import dicom
#f=[dicom.read_file(data_path+patients[0]+'/'+s) for s in os.listdir(data_path+patients[0])]
#print len(f)
#import matplotlib.pyplot as plt
#plt.imshow(f[0].pixel_array)
#plt.show()

#from joblib import Parallel, delayed
#import multiprocessing
#inputs = range(10)
#def processInput(i):
#    return i * i
#num_cores = multiprocessing.cpu_count()
#results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
#print num_cores
#print results
#
#
