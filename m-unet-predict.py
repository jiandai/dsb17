'''
ref https://www.kaggle.com/c/data-science-bowl-2017#tutorial
ver 20170322 by jian: test on without ROI
ver 20170323 by jian: clone the tutorial github repos
ver 20170324 by jian: merge tutorial py script together
ver 20170325 by jian: note a bug in dry run
ver 20170330 by jian: review output of pretained unet
ver 20170331 by jian: revamp using LUNA data,split the logic into two parts, pay more attention to numeric type & its conversion
ver 20170405 by jian: revamp
ver 20170407 by jian: save the extracted features, split the logic, forking
ver 20170408 by jian: modify for stage2
ver 20170409.1 by jian: take v2 features
ver 20170409.2 by jian: take batched 
final logloss: (198 * 0.611556059351 + 506 * 0.58916) / (198 + 506) = 0.5954588916924688
29 new Magpie Anil Thomas Shai Ronen 0.59545 2 13h
30 new UBC_LaunchPad Andr Anuar Yeraliyev Cheng Xie AndyQiu 0.59890 3 10h
among 2037 teams
as of 4/10 12:27am (30/2037~1.5%)
ver 20170410 by jian: compare stage2_features-v2.npy /w batch out

to-do:
'''

import pandas as pd
import numpy as np

#feature_array2 = np.load('stage2_features.npy')
#feature_array2 = np.concatenate([np.load('stage2_features-v2-b'+str(j)+'.npy') for j in range(-4,1)])
feature_array2 = np.concatenate([np.load('stage2_features-v4-b'+str(j)+'.npy') for j in range(1,6)])
#feature_array2_ = np.load('stage2_features-v2.npy')
#print feature_array2.shape
#print feature_array2_.shape
#diff = feature_array2 - feature_array2_
#print diff
#print diff.max(), diff.min()
#print np.histogram(diff)
# Completely identical


from classify_nodes import logloss,classifyData
from sklearn.metrics import classification_report

#feature_array = np.load('training_features-v2.npy')
#feature_array = np.concatenate([np.load('training_features-v2-b'+str(j)+'.npy') for j in range(8,22)])
feature_array = np.concatenate([np.load('training_features-v4-b'+str(j)+'.npy') for j in range(1,15)])
print feature_array.shape
labels_csv = pd.read_csv('../input/stage1_labels.csv', index_col='id')
batch_start=0
batch_end= feature_array.shape[0]
truth_metric = labels_csv.cancer[batch_start:batch_end]
rf,xg= classifyData(feature_array,truth_metric) #s5

#feature_array = np.concatenate([np.load('test_features-v2-b'+str(j)+'.npy') for j in range(-1,1)])
feature_array = np.concatenate([np.load('test_features-v4-b'+str(j)+'.npy') for j in range(1,3)])
print feature_array.shape
#feature_array_ = np.load('test_features-v2.npy')
#print feature_array_.shape
#diff = feature_array - feature_array_
#print diff
#print diff.max(), diff.min()
#print np.histogram(diff)
# Completely identical

stage1_solution = pd.read_csv('../input/stage1_solution.csv', index_col='id')
truth_metric =stage1_solution.cancer.values


test_csv = pd.read_csv('../input/stage1_sample_submission.csv', index_col='id')
stage2_csv = pd.read_csv('../input/stage2_sample_submission.csv', index_col='id')



def pred(clf,X,y=None,csv=None,outfile=None,validation=False):
	predicted = clf.predict(X)
	predicted_prob = clf.predict_proba(X)
	if validation:
		print 'log-loss=',logloss(y,predicted_prob[:,1])
		print predicted
		print y
		print classification_report(y,predicted, target_names=["No Cancer", "Cancer"])
	if csv is not None:
		csv.cancer = predicted_prob[:,1]
		#print csv
		csv.to_csv(outfile)

print 'all .5 log-loss=',logloss(truth_metric,test_csv.cancer.values)

# RF
pred(clf=rf,X=feature_array,y=truth_metric,csv=test_csv.copy(),outfile='stage1_submission_by_rf-v4.csv',validation=True)
pred(clf=rf,X=feature_array2,csv=stage2_csv.copy(),outfile='stage2_submission_by_rf-v4.csv')
# XG
pred(clf=xg,X=feature_array,y=truth_metric,csv=test_csv.copy(),outfile='stage1_submission_by_xg-v4.csv',validation=True)
pred(clf=xg,X=feature_array2,csv=stage2_csv.copy(),outfile='stage2_submission_by_xg-v4.csv')


