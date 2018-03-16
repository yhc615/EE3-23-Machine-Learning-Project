import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import numpy as np

from src.utils.data_utils import *

def print_metrics(classifier,X_in,Y_in,X_out,Y_out,preds_in,preds_out,verbose=False):
	class_names = ["Worst","Worse","Average","Better","Best"]
	if verbose:
		print("Confusion Matrix(Training Sample):\n")
		cnf_matrix = metrics.confusion_matrix(Y_in, preds_in)
		np.set_printoptions(precision=2)
		# Plot non-normalized confusion matrix
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
		# Plot normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
		plt.show()

		print('Mean squared error for training data:', metrics.mean_squared_error(Y_in, preds_in))	
		print("Training set score(sklearn):",classifier.score(X_in,Y_in))
		print("\nClassification Report(Training set):\n\n",metrics.classification_report(Y_in,preds_in))
		print("\n")
	
	print("Confusion Matrix(Testing Sample):\n")
	cnf_matrix = metrics.confusion_matrix(Y_out, preds_out)
	np.set_printoptions(precision=2)
	# Plot non-normalized confusion matrix
	# plt.figure()
	# plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
	plt.show()
	
	print('Mean squared error for test data:', metrics.mean_squared_error(Y_out, preds_out))
	print("Test set score(sklearn):",classifier.score(X_out,Y_out))#score meaning depends on classifier
	print("Classification Report(Test Sample):\n",metrics.classification_report(Y_out,preds_out))

def classify(datapath,v):#datapath: directory name of the datasets, (v)erbose: True or false 
	# Grab both wine datasets in one dataset
	concat_data = get_data(datapath)
	# Bag data to 5 scores 
	recode = {3:0, 4:0, 5:1, 6:2, 7:3, 8:4,9:4}
	concat_data['quality_c'] = bag_data(recode,concat_data,'quality')

	# Split up dataset 70/30 training,testing
	y_wine = concat_data['quality_c']
	X_wine = concat_data.drop(['quality_c','quality'], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=420)
	#==========================================================================================
	# Linear Regression Estimaro, OnevsRest classification
	clf = OneVsRestClassifier(estimator=linear_model.LinearRegression())
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*30+"\nLinear Regression:\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v)
	#==========================================================================================
	# Support Vector Estimator, OnevsRest classification
	clf = OneVsRestClassifier(estimator=svm.SVC(kernel='rbf',random_state=0))
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*30+"\nSVC :\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v)
	#==========================================================================================

if __name__ == '__main__':
	classify(str(sys.argv[1]),bool(sys.argv[2]))
