'''
'''
def get_one_scan(path):
	import os
	import dicom
	import numpy
	slices = [ dicom.read_file(path+'/'+s) for s in os.listdir(path) ]
	slices.sort(key=lambda x:x.InstanceNumber) 

	'''
	for slice in slices:
		# map from pixel space to RCS / DPCS
		#
		Xs0 = slice.ImagePositionPatient[0]
		Ys0 = slice.ImagePositionPatient[1]
		Zs0 = slice.ImagePositionPatient[2]
		#
		delta_r = slice.PixelSpacing[0]
		delta_c = slice.PixelSpacing[1]
		# row-axis direction, to time column change
		nX_c = slice.ImageOrientationPatient[0]
		nY_c = slice.ImageOrientationPatient[1]
		nZ_c = slice.ImageOrientationPatient[2]
		# column-axis direction, to time row change
		nX_r = slice.ImageOrientationPatient[3]
		nY_r = slice.ImageOrientationPatient[4]
		nZ_r = slice.ImageOrientationPatient[5]
		#
		Xs = numpy.empty((slice.Rows,slice.Columns))
		Ys = numpy.empty((slice.Rows,slice.Columns))
		Zs = numpy.empty((slice.Rows,slice.Columns))
		#
		for r in range(slice.Rows): 
			for c in range(slice.Columns):
				Xs[r,c] = Xs0 + nX_c*delta_c*c + nX_r*delta_r*r
				Ys[r,c] = Ys0 + nY_c*delta_c*c + nY_r*delta_r*r
				Zs[r,c] = Zs0 + nZ_c*delta_c*c + nZ_r*delta_r*r
	'''

	voxels = numpy.stack( [s.pixel_array for s in slices] )

	# conversion
	voxels = voxels.astype(numpy.int16)
	voxels[voxels == -2000] = 0
	
	# Compute the average z-spacing
	S = len(slices)
	SliceThickness = (slices[0].SliceLocation - slices[S-1].SliceLocation)/(S-1) # assume n>1

	# Convert to Hounsfield units (HU)
	for s in range(S):
		#del slices[s][slices[s].data_element("PixelData").tag] # no longer keep PixelData
		#slices[s].SliceThickness = SliceThickness
		
		intercept = slices[s].RescaleIntercept
		slope = slices[s].RescaleSlope
		if slope != 1:
			voxels[s] = slope * voxels[s].astype(numpy.float64)
			voxels[s] = voxels[s].astype(numpy.int16)
		voxels[s]+= numpy.int16(intercept)

	voxels = numpy.array(voxels, dtype=numpy.int16) 
	return voxels # return a 3-d numpy array

def get_all_scans(path):
	'''
	'''
	import os
	import dicom
	import numpy
	patients = os.listdir(path)
	all_scans=[]
	all_3d_volumes=[]
	for pat in patients[0:3]:



		# Some check on the most used features of one scan
		#
		all_scans = all_scans + [one_scan]
		all_3d_volumes = all_3d_volumes + [voxels]
	return (patients,all_scans,all_3d_volumes)

def resampling(scans,volumes,new_spacing=[1,1,1]):
	resampled_volumes = []
	for p in range(len(scans)):
		scan = scans[p]
		volume = volumes[p]
		old_shape = volume.shape
		import numpy
		spacing = numpy.array([scan[0].SliceThickness]+ scan[0].PixelSpacing,dtype=numpy.float32)
		reshape_factor = numpy.round(spacing/new_spacing * old_shape)/old_shape
		new_shape = reshape_factor * old_shape
		new_spacing = spacing / reshape_factor
		import scipy.ndimage
		resampled_volumes.append(scipy.ndimage.interpolation.zoom(volume,reshape_factor,mode='nearest'))
	return resampled_volumes


def normalize(image):
	MIN_BOUND = -1000.0
	MAX_BOUND = 400.0
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image>1] = 1.
	image[image<0] = 0.
	return image


def zero_center(image):
	PIXEL_MEAN = 0.25
	image = image - PIXEL_MEAN
	return image
