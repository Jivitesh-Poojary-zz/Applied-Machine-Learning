# Credit card fruad transaction data
# Undersampling - logistic regression - bagging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def data_out(o):
	import csv
	with open("output-RanFor.csv", "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(o)

def play(N,n,d):
	
	print("N=",N)
	print("n=",n)
	df = pd.read_csv('./trainccard.csv')
	test = pd.read_csv('./testccard.csv')
	data_true = df.loc[(df.Class==1)]
	data_false = df.loc[(df.Class==0)]

	train_sets=[]
	train_class_sets=[]
	models=[]
	y_score = []

	for i in range(N):
		rs=data_false.sample(n=775,replace=False)
		frames = [data_true,rs]
		train=pd.concat(frames)
		train_class = train[['Class']]
		train=train.drop('Class',1)
		train_sets.append(train)
		train_class_sets.append(train_class)

	for i in range(N):
		if d==0:
			clf = RandomForestClassifier(n_estimators=n)
		else:
			clf = RandomForestClassifier(n_estimators=n,max_depth=d)
		clf = clf.fit(train_sets[i],np.ravel(train_class_sets[i]))
		models.append(clf)

	ipreds=[]
	fpreds=[]
	test_p = test.drop('Class',1)
	
	#print(test_p)

	for i in range(N):
		t=models[i].predict(test_p)
		y_score.append(models[i].predict_proba(test_p))
		ipreds.append(t)
	
	print(y_score)

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

	print ("False negatives: ",len(FN)) 
	print ("False positives: ",len(FP) )
	acc=float(len(TN)+len(TP))/float((len(FN)+len(FP)+len(TN)+len(TP)))
	print ("Accuracy:",acc )
	prec =float(len(TP))/float(len(TP)+len(FP))
	rec = float(len(TP))/float(len(TP)+len(FN))
	print ("Precision",prec)
	print ("Recall",rec)
	print ("----------------------------------------------------------------------------")

####################################################################### 

	# setup plot details
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
	lw = 2
	n_classes = 2
	
	X_train = train_class_sets
	X_test = test_p
	#y_test = 
	y_score = fpreds
    
#	# Compute Precision-Recall and plot curve
#	precision = dict()
#	recall = dict()
#	average_precision = dict()
#	for i in range(n_classes):
#		precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
#		average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

#	# Compute micro-average ROC curve and ROC area
#	precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
#	average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
#
#	# Plot Precision-Recall curve
#	plt.clf()
#	plt.plot(recall[0], precision[0], lw=lw, color='navy',label='Precision-Recall curve')
#	plt.xlabel('Recall')
#	plt.ylabel('Precision')
#	plt.ylim([0.0, 1.05])
#	plt.xlim([0.0, 1.0])
#	plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
#	plt.legend(loc="lower left")
#	plt.show()
#
#	# Plot Precision-Recall curve for each class
#	plt.clf()
#	plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
#			 label='micro-average Precision-recall curve (area = {0:0.2f})'
#				   ''.format(average_precision["micro"]))
#				   
#	for i, color in zip(range(n_classes), colors):
#		plt.plot(recall[i], precision[i], color=color, lw=lw,
#				 label='Precision-recall curve of class {0} (area = {1:0.2f})'
#					   ''.format(i, average_precision[i]))
#
#	plt.xlim([0.0, 1.0])
#	plt.ylim([0.0, 1.05])
#	plt.xlabel('Recall')
#	plt.ylabel('Precision')
#	plt.title('Extension of Precision-Recall curve to multi-class')
#	plt.legend(loc="lower right")
#	plt.show()

####################################################################### 
 
	return N,n,d,len(FN),len(FP),len(TP),len(TN),acc, prec, acc

lm=[]

bags=[5,10,15,20,25,30,35,40,45,50]
depths=[0,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#TEST
l=[]
l=play(2,1,0)
lm.append(l)
#data_out(lm)


# for i in bags:
	# for j in range(11):
		# for k in depths:
			# if i>0 and j>0:
				# l=[]
				# l=play(i,j,k)
				# lm.append(l)
				# data_out(lm)

