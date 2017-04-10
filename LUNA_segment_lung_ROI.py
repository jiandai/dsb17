'''
forked from 
https://github.com/booz-allen-hamilton/DSB3Tutorial
as 
https://github.com/jiandai/DSB3Tutorial

hacked ver 20170323 by jian: 
	- use sample scan for DSB17 as input instead
	- node mask in this script is only resized / commented out entirely
	- merge two loops into one single loop
	- redefine the output as preprocessed figures
ver 20170324.1 by jian: use stage1 scan (or the resampled)
ver 20170324.2 by jian: spin off from the folk, and turn to a function / merge steps
ver 20170325 by jian: tune on whether to make the output a square and to resize to 512^2
ver 20170327.1 by jian: away from 512, test on "img[mask>0].shape[0]>0"
ver 20170327.2 by jian: output img in original size together with the box coordinates, mark the normalization
ver 20170331 by jian: prepare to use unet trained by full LUNA data 
ver 20170405 by jian: change default in segment_ROI to be consistent /w ../DSB3Tutorial/tutorial_code/LUNA_segment_lung_ROI.py

to-do: 
'''

def segment_ROI(img,normalize=True,keep_size=True,to_square=True,resizing=True):
	# Assume float64 for img
	from skimage import morphology
	from skimage import measure
	from skimage.transform import resize
	from sklearn.cluster import KMeans
	import numpy as np
	
	#res_x,res_y = img.shape[0],img.shape[1]
	res_x,res_y = 512,512

	#Standardize the pixel values
	if normalize:
		mean = np.mean(img)
		std = np.std(img)
		img = img-mean
		img = img/std
	# Find the average pixel value near the lungs
	# to renormalize washed out images
	#r1=100/512
	#r2=400/512
	middle = img[100:400,100:400] 
	#middle = img[res_x*r1:res_x*r2,res_y*r1:res_y*r2] 
	mean = np.mean(middle)  

	# To improve threshold finding, I'm moving the 
	# underflow and overflow on the pixel spectrum
	if normalize:
		max = np.max(img)
		min = np.min(img)
		img[img==max]=mean
		img[img==min]=mean

	#
	# Using Kmeans to separate foreground (radio-opaque tissue)
	# and background (radio transparent tissue ie lungs)
	# Doing this only on the center of the image to avoid 
	# the non-tissue parts of the image as much as possible
	#
	kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
	centers = sorted(kmeans.cluster_centers_.flatten())
	threshold = np.mean(centers)
	thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
	#
	# I found an initial erosion helful for removing graininess from some of the regions
	# and then large dialation is used to make the lung region 
	# engulf the vessels and incursions into the lung cavity by 
	# radio opaque tissue
	#
	eroded = morphology.erosion(thresh_img,np.ones([4,4]))
	dilation = morphology.dilation(eroded,np.ones([10,10]))
	#
	#  Label each region and obtain the region properties
	#  The background region is removed by removing regions 
	#  with a bbox that is to large in either dimnsion
	#  Also, the lungs are generally far away from the top 
	#  and bottom of the image, so any regions that are too
	#  close to the top and bottom are removed
	#  This does not produce a perfect segmentation of the lungs
	#  from the image, but it is surprisingly good considering its
	#  simplicity. 
	#
	labels = measure.label(dilation)
	label_vals = np.unique(labels)
	regions = measure.regionprops(labels)
	good_labels = []
	for prop in regions:
		B = prop.bbox
		if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
			good_labels.append(prop.label)
	mask = np.ndarray([res_x,res_y],dtype=np.int8)
	mask[:] = 0
	#
	#  The mask here is the mask for the lungs--not the nodes
	#  After just the lungs are left, we do another large dilation
	#  in order to fill in and out the lung mask 
	#
	for N in good_labels:
		mask = mask + np.where(labels==N,1,0)
	mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
	#imgs_to_process[i] = mask
	
	
	new_size = [res_x,res_y]   # we're scaling back up to the original size of the image
	img= mask*img          # apply lung mask
	#
	# renormalizing the masked image (in the mask region)
	#

	if normalize and img[mask>0].shape[0]>0:
		new_mean = np.mean(img[mask>0])  
		new_std = np.std(img[mask>0])
		#
		#  Pulling the background color up to the lower end
		#  of the pixel range for the lungs
		#
		old_min = np.min(img)       # background color
		img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
		img = img-new_mean
		img = img/new_std

	#make image bounding box  (min row, min col, max row, max col)
	labels = measure.label(mask)
	regions = measure.regionprops(labels)
	#
	# Finding the global min and max row over all regions
	#
	min_row = res_x
	max_row = 0
	min_col = res_y
	max_col = 0
	for prop in regions:
		B = prop.bbox
		if min_row > B[0]:
			min_row = B[0]
		if min_col > B[1]:
			min_col = B[1]
		if max_row < B[2]:
			max_row = B[2]
		if max_col < B[3]:
			max_col = B[3]
	width = max_col-min_col
	height = max_row - min_row
	if to_square:
		if width > height:
			max_row=min_row+width
		else:
			max_col = min_col+height
	# 
	# cropping the image down to the bounding box for all regions
	# (there's probably an skimage command that can do this in one line)
	# 
	#mask =  mask[min_row:max_row,min_col:max_col]
	if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
		new_img=None
	elif keep_size:
		new_img=img
	else:
		new_img = img[min_row:max_row,min_col:max_col] 
		# moving range to -1 to 1 to accomodate the resize function
		mean = np.mean(new_img)
		new_img = new_img - mean
		min = np.min(new_img)
		max = np.max(new_img)
		new_img = new_img/(max-min)
		if resizing:
			new_img = resize(new_img,[res_x,res_y]) 
	#print min_row,max_row,min_col,max_col
	return new_img, (min_row,max_row,min_col,max_col)



def debugPlot(x):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	cax=ax.imshow(x,cmap='gray')
	cbar = fig.colorbar(cax)
	plt.show()

