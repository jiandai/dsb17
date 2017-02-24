#%%
# -*- coding: utf-8 -*-
"""
version 20170116 by Jian:
follow:
    https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
ref:
    https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/

version 20170131 by Jian: revisit
version 20170206 by Jian: revisit

version 20170221 by Jian: more on visual 
version 20170222 by Jian: batch process, metadata DB
version 20170223 by Jian: metadata at patient level, slide level, and unhashable types

to-do normalization, packaging, OOing
"""



PWD='..'
INPUT_FOLDER=PWD+'/sample_images/'
import os
patients = os.listdir(INPUT_FOLDER)
print(patients)

import dicom
allScan=[[dicom.read_file(INPUT_FOLDER+pat+'/'+s) for s in os.listdir(INPUT_FOLDER+pat)] for pat in patients]


#%%
fstfst = allScan[0][0]
#%%
scdfst = allScan[1][0]

#%%
print(fstfst.dir())
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%


#%%
rec=dict()
pos=dict()
for name in fstfst.dir():
    if name!='PixelData':
        print(name+'[')
        print(fstfst.data_element(name).tag)
        print(fstfst.data_element(name).name)
        print(fstfst.data_element(name).VR)
        print(fstfst.data_element(name).VM)
        print(fstfst.data_element(name).value)
        print(']')
        rec[name] = fstfst.data_element(name).value
        pos[name] = fstfst.data_element(name).tag
    else:
        print(type(fstfst.PixelData))

#%%
import pandas as pd
#%%
pd.DataFrame([{'a':1,'b':2}]).append(pd.DataFrame([{'a':4,'b':5}]))
#%%
pd.DataFrame([{'b':2}]).append(pd.DataFrame([{'a':'bb','b':5}]))
#%%
pd.DataFrame([{'a':'aa','b':2}]).append(pd.DataFrame([{'b':5}]))
#%%
pd.DataFrame([rec]).append(pd.DataFrame([rec])).shape
#%%
metaDB0 = pd.DataFrame()
colDB0 = pd.DataFrame()
for sld in allScan[0]:
    red=dict()
    for name in sld.dir():
        if name!='PixelData':
            red[name] = sld.data_element(name).value
            colDB0 = colDB0.append(pd.DataFrame([{'col':name,'pos':sld.data_element(name).tag}]))
    metaDB0 = metaDB0.append(pd.DataFrame([red]))
#%%
metaDB0.shape
#%%
colDB0
#%%
colDB0.drop_duplicates().shape
#%%
bs = set(fstfst.dir())
allCol=bs
cnt = 0
metaDB = pd.DataFrame()
colDB=pd.DataFrame()
for patScan in allScan:
    for sld in patScan:
        cnt += 1
        s=set(sld.dir())
        diff=s.symmetric_difference(bs)
        rec=dict()
        for name in sld.dir():
            if name != 'PixelData':
                rec[name] = sld.data_element(name).value
                colDB = colDB.append(pd.DataFrame([{'col':name,'pos':sld.data_element(name).tag}]))
        metaDB = metaDB.append(pd.DataFrame([rec]))
        if len(diff)>0:
            print(sld.PatientID)
            print(sld.InstanceNumber)
            allCol=allCol.union(s)
            #print(diff)
        #for it in sld.dir():
        #    print(it)
#%%
print(cnt)
#%%
metaDB.shape
#%%
colDB.shape
#%%
colDB.drop_duplicates().shape
#%%
colDB1=colDB.drop_duplicates()
#%%
colDB1 = colDB1.sort_values(['pos'])



#%%
'''
Here we try two different ways to define the preferred order of the meta data fields
'''
#%%
preferredOrder = colDB1.col
#%%
''' 
preferredOrder = ['PatientID',
'PatientName',
'InstanceNumber',
'AcquisitionNumber',
'BitsAllocated',
'BitsStored',
'BurnedInAnnotation',
'Columns',
'Rows',
'FrameOfReferenceUID',
'HighBit',
'ImageOrientationPatient',
'ImagePositionPatient',
'KVP',
'LargestImagePixelValue',
'LongitudinalTemporalInformationModified',
'Modality',
'PatientBirthDate',
'PatientOrientation',
'PhotometricInterpretation',
'PixelAspectRatio',
'PixelPaddingValue',
'PixelRepresentation',
'PixelSpacing',
'PlanarConfiguration',
'PositionReferenceIndicator',
'RescaleIntercept',
'RescaleSlope',
'RescaleType',
'SOPClassUID',
'SOPInstanceUID',
'SamplesPerPixel',
'SeriesDescription',
'SeriesInstanceUID',
'SeriesNumber',
'SliceLocation',
'SmallestImagePixelValue',
'SpecificCharacterSet',
'StudyInstanceUID',
'WindowWidth',
'WindowCenter',
'WindowCenterWidthExplanation']
# minus 'PixelData',
'''
#%%
metaDB = metaDB [list(preferredOrder)]
#%% 
metaDB = metaDB.sort_values(['PatientID','InstanceNumber'])
#%%
#metaDB.to_csv('sample_images_dicom_metadata.csv',index=False) # use manually defined preferred order
metaDB.to_csv('sample_images_dicom_metadata_v1.2.csv',index=False) # use automatically created preferred order by dicom tags



#%%
# some series cannot use unique method due the type is not HASHABL
# lesson is DF may not be the best data structure for some operations to store DICOM metadata
#%%
N=metaDB.PatientID.unique().shape[0]
#%%
patLevelCol=dict()
slideLevelCol=dict()
unHashableCol=dict()
#%%
for c in metaDB.columns[:]:    
    try:
        cnt=metaDB[c].unique().shape[0]
        print(cnt)
        if cnt>N:
            slideLevelCol[c] = cnt
            print(len(slideLevelCol))
        else:
            patLevelCol[c] = cnt
            print(len(patLevelCol))           
    except TypeError:
        unHashableCol[c] = type(metaDB[c].iloc[0])
        #print(c+'='+metaDB[c].apply(lambda x:str(x)).unique().shape[0])

#%%
# Put patient level tags in one dataset, and slide level + unhashables in another one
metaDB[[k for k,v in patLevelCol.items()]].drop_duplicates().shape
#%%
patLevelDB = metaDB[[k for k,v in patLevelCol.items()]].drop_duplicates()
#%%
slideLevelDB = metaDB[['PatientID']+[k for k,v in slideLevelCol.items()] + [k for k,v in unHashableCol.items()]]
#%%
patLevelDB.columns.intersection(slideLevelDB.columns) # PatientID only

#%%
# Check the dup in patient level DB
patLevelDB.groupby(['PatientID']).count()
#%%
patLevelDB.groupby(['PatientID']).size()
#%%
patLevelDB[patLevelDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068'].shape
patLevelDB[patLevelDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068'].to_csv('dup.csv')
patLevelDB.drop('AcquisitionNumber',1).drop_duplicates().shape
# So the root of the dup is for one particular patient having 5 "AcquisitionNumber" in ths slides

metaDB[metaDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068'].shape
dupPat = metaDB[metaDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068']
dupPat.shape
dupPat.groupby(['AcquisitionNumber']).size()
'''
AcquisitionNumber
1    240
2      5
3      5
4     10
5     20
dtype: int64
'''
# Move /w Instance number
dupPat[['InstanceNumber','AcquisitionNumber']]


# output:


'''
=> Pixel data
most interested fields:
Instance Number
Image Position
Image Orientation
Slice Location
Pixel Spacing 
Rescale Intercept
Rescale Slope

'''

quit()

import matplotlib.pyplot as plt


# show individual slide
def showSlide(patId, slideNum):
	slices=[dicom.read_file(INPUT_FOLDER+patients[patId]+'/'+s) for s in os.listdir(INPUT_FOLDER+patients[patId])]
	print(slices[slideNum])
	plt.imshow(slices[slideNum].pixel_array)
	plt.title('pat '+str(patId)+': slide'+str(slideNum) + ' (without normalization)')
	plt.colorbar()
	plt.show()


# show a whole scan
def showScan(patId=0): #134
	slices=[dicom.read_file(INPUT_FOLDER+patients[patId]+'/'+s) for s in os.listdir(INPUT_FOLDER+patients[patId])]
	slices.sort(key=lambda x: x.InstanceNumber)
	#print([(s.InstanceNumber,s.SliceLocation) for s in slices])
	print(len(slices))
	nrow=len(slices) // 10 + (0 if len(slices) % 10 ==0 else 1)
	ncol=10
	for i in range(0, len(slices)):
		fig=plt.subplot(nrow,ncol,1 + i)
		fig.set_title(str(i))
		plt.imshow(slices[i].pixel_array)
	plt.show()
'''
showSlide(9,11)
showSlide(9,11)
showSlide(14,51)
'''
for pat in [3,5,9]: #range(len(patients)):
	print(pat)
	showScan(patId=pat) #134


quit()


# for len being 110
#nrow=10
#ncol=11
# for len being 133







#for i in range(25, 50):
#	plt.subplot(5,5,i-24)
#	plt.imshow(slices[i].pixel_array)
#plt.show()


quit()

import numpy as np
slice_thickness=np.abs(np.array([slices[j-1].SliceLocation-slices[j].SliceLocation for j in range(1,len(slices))]).mean())

image = np.stack([s.pixel_array for s in slices])
image=image.astype(np.int16) # from nint16
image[image==-2000]=0

intercept = slices[0].RescaleIntercept
slope = slices[0].RescaleSlope

if slope != 1:
	image = splope * image.astype(np.float64)
	image = image.astype(np.int16)
image += np.int16(intercept)
image=image.astype(np.int16)

plt.hist(image.flatten(),bins=100)
plt.xlabel('Housfield Units (HU)')
plt.ylabel('Frequency')
plt.show()

plt.imshow(image[1],cmap=plt.cm.gray)







#%%
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p = image.transpose(2,1,0)
p = p[:,:,::-1]
verts, faces = measure.marching_cubes(p,100)
#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.1)
face_color = [0.5, 0.5, 1]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)
ax.set_xlim(0, p.shape[0])
ax.set_ylim(0, p.shape[1])
ax.set_zlim(0, p.shape[2])
plt.show()


quit()




[slices[0].SliceThickness]+ slices[0].PixelSpacing
spacing = np.array([slices[0].SliceThickness]+ slices[0].PixelSpacing,dtype=np.float32)
resize_factor=spacing/[1,1,1]
new_shape=np.round(image.shape * resize_factor)
real_resize_factor=new_shape/image.shape
spacing /real_resize_factor

PhysicalSpans = image.shape * spacing # physical sizes
np.round(PhysicalSpans / [1,1,1]) == new_shape
PhysicalSpans / new_shape - spacing /real_resize_factor




import scipy.ndimage
image1=scipy.ndimage.interpolation.zoom(image,real_resize_factor,mode='nearest')
image.shape
image1.shape
# doesn't work in pycharm
#%matplotlib inline


###########################################################################################

from skimage import morphology
import pandas as pd




# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#%%
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16) 
#%%
first_patient = load_scan(INPUT_FOLDER + patients[0])
#%%
first_patient_pixels = get_pixels_hu(first_patient)
#%%
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
#%%
#%%
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing
#%%
pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)
#%%
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

#%%
plot_3d(pix_resampled, 400)
#%%
#PathDicom = 'C:/Users/daij12/Documents/Analysis/ML/kaggle/data-science-bowl-2017/sample_images/00cba091fa4ad62cc3200a657aeb957e/'
#lstFilesDCM = []  # create an empty list
#for dirName, subdirList, fileList in os.walk(PathDicom):
#    for filename in fileList:
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
#            lstFilesDCM.append(os.path.join(dirName,filename))
#%%
#RefDs = dicom.read_file(lstFilesDCM[0])
#%%
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
#ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
#%% thickness not available
# Load spacing values (in mm)
#ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
