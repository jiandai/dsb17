'''
forked from
https://github.com/booz-allen-hamilton/DSB3Tutorial
as
https://github.com/jiandai/DSB3Tutorial
hacked ver 20170323 by jian:
	- rewire the input/output from LUNA_segment_lung_ROI.py
	- use pretrained unet in LUNA_train_unet.py
	- create feature /w one sample file
ver 20170324 by jian: spin off from folked tutoril
to-do:
=> rewire truthdata

note:
/gne/home/daij12/.local/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: 
This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and 
functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module 
will be removed in 0.20. "This module will be removed in 0.20.", DeprecationWarning)

'''

def getRegionFromMap(slice_npy):
    import numpy as np
    from skimage import measure
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    label_image = measure.label(thr)
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions

def getRegionMetricRow(seg):
    import numpy as np
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    nslices = seg.shape[0]
    
    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    
    areas = []
    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1
    if totalArea>0:
        weightedX = weightedX / totalArea 
        weightedY = weightedY / totalArea
    if numNodes>0:
        avgArea = totalArea / numNodes
        avgEcc = avgEcc / numNodes
        avgEquivlentDiameter = avgEquivlentDiameter / numNodes
        stdEquivlentDiameter = np.std(eqDiameters)
    if len(areas)>0: 
        maxArea = max(areas)
    
    
    numNodesperSlice = numNodes*1. / nslices
    
    
    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])











# usage: python classify_nodes.py nodes.npy 




def createFeatureDataset(nodfiles=None):
    import pickle
    if nodfiles == None:
        # directory of numpy arrays containing masks for nodules
        # found via unet segmentation
        noddir = "/training_set/" 
        nodfiles = glob(noddir +"*npy")
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    truthdata = pickle.load(open("truthdict.pkl",'r'))
    numfeatures = 9
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))
    
    for i,nodfile in enumerate(nodfiles):
        patID = nodfile.split("_")[2]
        truth_metric[i] = truthdata[int(patID)]
        feature_array[i] = getRegionMetricRow(nodfile)
    
    np.save("dataY.npy", truth_metric)
    np.save("dataX.npy", feature_array)

def logloss(act, pred):
    import scipy as sp
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def classifyData(X,Y):
    import numpy as np
    from sklearn import cross_validation
    from sklearn.cross_validation import StratifiedKFold as KFold
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier as RF
    import xgboost as xgb

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # All Cancer
    print "Predicting all positive"
    y_pred = np.ones(Y.shape)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # No Cancer
    print "Predicting all negative"
    y_pred = Y*0
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = xgb.XGBClassifier(objective="binary:logistic")
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss",logloss(Y, y_pred))

if __name__ == "__main__":
    from sys import argv  
    
    #getRegionMetricRow(argv[1:])
    #print getRegionMetricRow('masksTestPredicted.npy')
    print getRegionMetricRow('masksTestPredicted-test-2.npy')
'''
[  2.16875000e+02   2.57900000e+03   6.90278760e-01   9.16564621e+00
   1.38609075e+01   1.62084726e+02   3.15923055e+02   1.60000000e+01
   5.33333333e+00]
'''
    #classifyData()
