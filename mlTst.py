"""
version 20170404.1 by jian: modified from a ML test script, when run from rescomp4, no need to explicitly load local sklearn for "model_selection"
version 20170404.2 by jian: same all negative prediction
"""
import numpy as np
X = np.load('seg-out-feature-array.npy')
print X.shape

import pandas as pd
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
Y = labels_csv.values
print Y.shape
y = Y.reshape(Y.shape[0])

from sklearn import model_selection
seed=7
X_tr,X_tt,y_tr,y_tt = model_selection.train_test_split(X,y,test_size=.3,random_state=seed)
print( X_tr.shape)
print( X_tt.shape)
print( y_tr.shape)
print( y_tt.shape)

from sklearn import svm
model = svm.SVC()
model.fit(X_tr,y_tr)
yhat_tt = model.predict(X_tt)



from sklearn import metrics
print metrics.accuracy_score(y_tt,yhat_tt)
print(metrics.classification_report(y_tt, yhat_tt))
print metrics.log_loss(y_tt,yhat_tt)
print(metrics.confusion_matrix(y_tt, yhat_tt))
'''
(1397, 9)
(1397, 1)
(977, 9)
(420, 9)
(977,)
(420,)
0.752380952381

/gne/home/daij12/.local/lib/python2.7/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.

  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       0.75      1.00      0.86       316
          1       0.00      0.00      0.00       104

avg / total       0.57      0.75      0.65       420

8.55245891684
[[316   0]
 [104   0]]
'''


quit()

seed=13
FOLD=10
model = svm.SVC()
kfold=model_selection.KFold(n_splits=FOLD,random_state=seed)
cv_results = model_selection.cross_val_score(model,X_tr,y_tr,cv=kfold,scoring='accuracy')
print(cv_results)
cv_results = model_selection.cross_val_score(model,X_tr,y_tr,cv=kfold,scoring='log_loss')
print(cv_results)

quit()





def algoTst(model,X_tr,y_tr):
	model.fit(X_tr,y_tr)
	yhat_tt=model.predict(X_tt)
	print(metrics.accuracy_score(y_tt, yhat_tt))

algoTst(model = svm.SVC(),X_tr=X,y_tr=Y)

quit()
def cvTst(model,FOLD=3):
	seed=11
	kfold=model_selection.KFold(n_splits=FOLD,random_state=seed)
	cv_results = model_selection.cross_val_score(model,X_tr,y_tr,cv=kfold,scoring='accuracy')

cvTst(model = svm.SVC())








import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')

#dataset = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data',sep='\s+',header=None,engine='python')
#dataset = pd.read_table('data/semeion.data.txt',sep='\s+',header=None,engine='python')
dataset = pd.read_table('data/semeion.data.txt',sep='\s+',header=None)
# col 0-255: data
# col 256-265: one-hot label


X,Y = dataset[list(range(256))],dataset[list(range(256,266))] # modified on pc by adding list
#print X.shape
#print Y.shape

#import matplotlib.pyplot as plt
#plt.imshow(X.iloc[19].values.reshape((16,16),order='C'),cmap='hot')
#plt.show()
#plt.imshow(X.iloc[20].values.reshape((16,16)),cmap='gray')
#plt.show()
#plt.imshow(X.iloc[21].values.reshape((16,16)))
#plt.show()
#plt.imshow(X.iloc[22].values.reshape((16,16)))
#plt.show()

#print Y.sum(axis=1)


# convert 1-hot code to a single variable
y=[]
for r in range(Y.shape[0]):
	for c in range(256,266):
		if Y[c].iloc[r]==1:
			#print r,c-256
			y.append(c-256)
y=np.array(y)







from sklearn import linear_model
algoTst(model = linear_model.LogisticRegression())
cvTst(model = linear_model.LogisticRegression())




from sklearn import neighbors
algoTst(model = neighbors.KNeighborsClassifier())
cvTst(model = neighbors.KNeighborsClassifier())

from sklearn import tree
algoTst(model = tree.DecisionTreeClassifier())
cvTst(model = tree.DecisionTreeClassifier())

from sklearn import naive_bayes
algoTst(model = naive_bayes.GaussianNB())
cvTst(model = naive_bayes.GaussianNB())

from sklearn import discriminant_analysis
algoTst(model = discriminant_analysis.LinearDiscriminantAnalysis())
cvTst(model = discriminant_analysis.LinearDiscriminantAnalysis())


