# Credit card fruad transaction data
# Undersampling - logistic regression - bagging

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def data_out(o):
	import csv
	with open("output-BaggedKNN.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(o)

def play(N,n,alpha):
	
	print("N=",N)
	print("n=",n)
	df = pd.read_csv('./trainccard.csv')
	test = pd.read_csv('./testccard.csv')
	data_true = df.loc[(df.Class==1)]
	data_false = df.loc[(df.Class==0)]

	train_sets=[]
	train_class_sets=[]
	models=[]

	for i in range(N):
		rs=data_false.sample(n=775,replace=False)
		frames = [data_true,rs]
		train=pd.concat(frames)
		train_class = train[['Class']]
		train=train.drop('Class',1)
		train_sets.append(train)
		train_class_sets.append(train_class)

	for i in range(N):
		if alpha==0:
			clf = BaggingClassifier(KNeighborsClassifier(),n_estimators=n)
		else:
			clf = BaggingClassifier(KNeighborsClassifier(),n_estimators=n)
		clf = clf.fit(train_sets[i],np.ravel(train_class_sets[i]))
		models.append(clf)

	ipreds=[]
	fpreds=[]
	test_p = test.drop('Class',1)

	for i in range(N):
		t=models[i].predict(test_p)
		ipreds.append(t)

	for j in range(len(test_p)):
		res=0
		for i in range(N):
			if ipreds[i][j]==1:
				res=1
		fpreds.append(res)

	test['Prob']=fpreds

	FN = test.loc[(test.Class==1)&(test.Prob!=1)]
	FP = test.loc[(test.Class==0)&(test.Prob!=0)]
	TP = test.loc[(test.Class==1)&(test.Prob==1)]
	TN = test.loc[(test.Class==0)&(test.Prob==0)]

	print("False negatives:",len(FN)) 
	print("False positives: ",len(FP) )
	acc=float(len(TN)+len(TP))/float((len(FN)+len(FP)+len(TN)+len(TP)))
	print("Accuracy:",acc) 
	print("----------------------------------------------------------------------------")
	return N,n,alpha,len(FN),len(FP),len(TP),len(TN),acc

lm=[]

bags=[5,10,15,20,25,30,35,40,45,50]
depths=[0,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
rates=[0,0.1,0.2,0.3]
iter=[1]


for i in bags:
	for j in range(11):
		for k in iter:
			if i>0 and j>0:
				l=[]
				l=play(i,j,k)
				lm.append(l)
				data_out(lm)

