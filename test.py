"""
version 20170116 by Jian:
follow:
    https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

version 20170131 by Jian: revisit
version 20170206 by Jian: revisit

version 20170221 by Jian: more on visual 
version 20170222 by Jian: batch process, metadata DB
version 20170223.1 by Jian: metadata at patient level, slice level, and unhashable types
version 20170223.2 by Jian: refine unhashable extraction
version 20170224.1 by jian: (partial) packaging, pixel to volume
version 20170224.2 by jian: more on DICOM, best ref:
http://nipy.org/nibabel/dicom/dicom.html
http://nipy.org/nibabel/dicom/dicom_orientation.html

version 2017025.1 by jian: dicom geometry
version 2017025.2 by jian: convert to haunsfield unit, resampling
to-do accelerate resampling, *OOing
more ref:
https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial
https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
"""
'''
import numpy as np
resampledPixels = np.load('resampledPixels.npz')
print(resampledPixels.keys())
print(resampledPixels['arr_0'].shape)
print(resampledPixels['arr_0'][0].shape)
print(resampledPixels['arr_0'][1].shape)
print(resampledPixels['arr_0'][2].shape)
import matplotlib.pyplot as plt
plt.hist(resampledPixels['arr_0'][0].flatten(),bins=100)
plt.show()
plt.hist(resampledPixels['arr_0'][1].flatten(),bins=100)
plt.show()
plt.hist(resampledPixels['arr_0'][2].flatten(),bins=100)
plt.show()
quit()
'''
PWD='..'
import os
INPUT_FOLDER= os.path.join(PWD,'sample_images')
from dicom_batch import get_all_scans
patients,allScan,allPixels = get_all_scans(INPUT_FOLDER)




from dicom_batch import resampling
resampledPixels = resampling(allScan,allPixels)

print(allPixels[0].shape)
#(134, 512, 512)
print(resampledPixels[0].shape)
#(335, 306, 306)
print(len(resampledPixels))
import numpy as np
np.savez('resampledPixels.npz',resampledPixels)




quit()    
# Per Guido Zuidhof's recommendation, the following two steps should be carried out prior to the training
from dicom_batch import normalize
normalizedPixels = [normalize(p) for p in resampledPixels]
from dicom_batch import zero_center
zerocenteredPixels = [zero_center(p) for p in normalizedPixels]


plt.imshow(normalizedPixels[0][0])
plt.colorbar()
plt.show()
plt.imshow(zerocenteredPixels[0][0])
plt.colorbar()
plt.show()
quit()




# raw value in the scan
from scipy.stats import itemfreq
frq = itemfreq(fstscan)
print(frq)
# Have to be sorted before present



quit()


print(fstscan[0][0,0])
print(fstscan[0][233,511-175])
print(fstscan[0][275,511-256]) # to match /w Mango output as Mango's X direction is per patient's view
'''
0,0,-3024,-2000
175,233,-202,822
256,275,298,1322
998
1344
-2000
1082
1004
'''

fig,axes = plt.subplots(1,3)
axes[0].imshow(fstscan[133], cmap="gray",origin='lower')
axes[1].imshow(fstscan[123])
axes[2].imshow(fstscan[100,:,:])
plt.suptitle('sample slices')
plt.show()
plt.imshow(fstscan[:,255,:])
plt.show()
plt.imshow(fstscan[:,:,255])
plt.show()
quit()
#%%
# use first slice of 1st and 2nd patient as example
fstfst = allScan[0][0]
'''
name=xx
print(name+'[')
print(fstfst.data_element(name).tag)
print(fstfst.data_element(name).name)
print(fstfst.data_element(name).VR)
print(fstfst.data_element(name).VM)
print(fstfst.data_element(name).value)
'''
scdfst = allScan[1][0]




#%%
# handle one patient or one scan
#%%
import pandas as pd
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

#%%
# handle all scans
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
            allCol=allCol.union(s)
            #print(diff)
        #for it in sld.dir():
        #    print(it)
#%%
colDB1=colDB.drop_duplicates()
colDB1 = colDB1.sort_values(['pos'])



#%%
preferredOrder = colDB1.col
# minus 'PixelData',
#%%
# Reorder by preferred order:
metaDB = metaDB [list(preferredOrder)]
#%%
# Use the key to sort
metaDB = metaDB.sort_values(['PatientID','InstanceNumber'])
#%%
#metaDB.to_csv('sample_images_dicom_metadata.csv',index=False) # use manually defined preferred order
metaDB.to_csv('sample_images_dicom_metadata_v1.2.csv',index=False) # use automatically created preferred order by dicom tags

#%%
# some series cannot use unique method due the type is not HASHABL
#%%
N=metaDB.PatientID.unique().shape[0]
#%%
patLevelCol=[]
sliceLevelCol=[]
#%%
for c in metaDB.columns[:]:    
    try:
        cnt=metaDB[c].unique().shape[0]
    except TypeError:
        metaDB[c] = metaDB[c].apply(str)
        cnt=metaDB[c].unique().shape[0]
    finally:
        if cnt>N:
            sliceLevelCol.append(c)
        else:
            patLevelCol.append(c)

#%%
patLevelDB = metaDB[patLevelCol].drop_duplicates()
#%%
sliceLevelDB = metaDB[['PatientID','AcquisitionNumber']+sliceLevelCol]
#%%
patLevelDB.columns.intersection(sliceLevelDB.columns) # PatientID only

#%%
# Check the dup in patient level DB
patLevelDB.groupby(['PatientID']).size()
#%%
# Dup case check
patLevelDB[patLevelDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068'].shape
patLevelDB[patLevelDB.PatientID == '0bd0e3056cbf23a1cb7f0f0b18446068'].to_csv('dup.csv',index=False)
patLevelDB.drop('AcquisitionNumber',1).drop_duplicates().shape
# So the root of the dup is for one particular patient having 5 "AcquisitionNumber" in ths slices

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

patLevelDB.to_csv('patient-level-non-pixel-meta-data.csv',index=False)
sliceLevelDB.to_csv('slice-level-non-pixel-meta-data.csv',index=False)









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
'''
for pat in [3,5,9]: #range(len(patients)):
	print(pat)
	showScan(patId=pat) #134


quit()






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




###########################################################################################

from skimage import morphology
import pandas as pd





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
#PathDicom = '../sample_images/00cba091fa4ad62cc3200a657aeb957e/'
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
