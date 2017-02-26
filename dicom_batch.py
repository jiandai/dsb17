'''
'''
def get_all_scans(path):
	'''
	'''
	import os
	import dicom
	import numpy
	patients = os.listdir(path)
	all_scans=[]
	all_3d_volumes=[]
	for pat in patients[0:1]:
		# one scan
		one_scan = []
		for s in os.listdir(path+'/'+pat):
			# one slice
			slice = dicom.read_file(path+'/'+pat+'/'+s)
			slice.FileName = s
			one_scan.append(slice)
		# Sort by the z-order
		one_scan.sort(key=lambda x:x.InstanceNumber) 
		pixel_stack =[]
		for slice in one_scan:
			pixel_array = slice.pixel_array

			pixel_stack.append(pixel_array)
			'''
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

		voxels = numpy.stack(pixel_stack)
		# normalization
		voxels = voxels.astype(numpy.int16)
		voxels[voxels == -2000] = 0
    		# Convert to Hounsfield units (HU)
		intercept = one_scan[0].RescaleIntercept
		slope = one_scan[0].RescaleSlope
		if slope != 1:
			voxels = slope * voxels.astype(numpy.float64)
			voxels = voxels.astype(numpy.int16)
		voxels+= numpy.int16(intercept)
		voxels = numpy.array(voxels, dtype=numpy.int16) 
    

		# Some check on the most used features of one scan

		n = len(one_scan)
		print(n)
		print(one_scan[n-1].Rows)
		print(one_scan[n-1].Columns)

		print(one_scan[n-1].InstanceNumber)
		print(one_scan[0].SliceLocation)
		print(one_scan[n-1].SliceLocation)
		print(one_scan[0].SliceLocation - one_scan[n-1].SliceLocation)


		print(one_scan[n-1].ImagePositionPatient) # physical coordinate of top left voxel
		print(one_scan[n-1].PixelSpacing)
		print(one_scan[n-1].ImageOrientationPatient) #

		print(one_scan[n-1].RescaleIntercept)
		print(one_scan[n-1].RescaleSlope)


		all_scans = all_scans + [one_scan]
		all_3d_volumes = all_3d_volumes + [voxels]
	return (patients,all_scans,all_3d_volumes)

