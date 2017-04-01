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
def tstPlot(a,b,n):
	fig,ax = plt.subplots(1,2,figsize=[5,5])
	ax[0].imshow(a[n],cmap='gray')
	ax[1].imshow(b[n],cmap='gray')
	plt.show()

scan,segs = get_pair(9)
tstPlot(scan,segs,44)


quit()
