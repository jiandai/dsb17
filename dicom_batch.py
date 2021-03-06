'''
ref: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
'''
def get_one_scan(path,resampling=True,new_spacing=[1,1,1]):
	import os
	import dicom
	import numpy
	slices = [ dicom.read_file(path+'/'+s) for s in os.listdir(path) ]
	#slices.sort(key=lambda x:x.InstanceNumber) 
	slices.sort(key=lambda x:x.ImagePositionPatient[2]) 

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
	#SliceThickness = (slices[0].SliceLocation - slices[S-1].SliceLocation)/(S-1) # assume n>1

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
	SliceThickness = get_thickness(slices)
	print 'raw slice-thickness=',SliceThickness
	if SliceThickness<0:
		voxels = voxels[::-1,:,:]
		SliceThickness = - SliceThickness
	spacing = numpy.array([SliceThickness]+ slices[0].PixelSpacing,dtype=numpy.float32)
	if resampling:
		voxels = resampling_one(voxels,spacing,new_spacing=new_spacing)
	return voxels, spacing # return a 3-d numpy array



def get_thickness(scan):
	S = len(scan)
	return (-scan[0].ImagePositionPatient[2] + scan[S-1].ImagePositionPatient[2])/(S-1) # assume n>1



def resampling_one(volume,spacing,new_spacing=[1,1,1]):
	import numpy
	# If assume SliceThickness is defined per slice
	old_shape = volume.shape
	reshape_factor = numpy.round(spacing/new_spacing * old_shape)/old_shape
	new_shape = reshape_factor * old_shape
	new_spacing = spacing / reshape_factor
	import scipy.ndimage
	return scipy.ndimage.interpolation.zoom(volume,reshape_factor,mode='nearest')

def largest_label_volume(im, bg=-1):
    import numpy as np
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True,dilating=True,dilation_size=5):
    import numpy as np
    from skimage import measure    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    from skimage import morphology
    binary_image = morphology.dilation(binary_image,np.ones([2*dilation_size,2*dilation_size,2*dilation_size]))
    #binary_image = morphology.dilation(binary_image,morphology.ball(dilation_size)) 
    return binary_image


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
		resampled_volumes.append( resampling_one(scan=scans[p],volume=volumes[p],new_spacing=new_spacing) )

	return resampled_volumes





