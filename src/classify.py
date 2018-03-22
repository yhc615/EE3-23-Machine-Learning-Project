import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn import neural_network
from sklearn import svm
from sklearn import metrics
import numpy as np

from src.utils.data_utils import *

def print_metrics(classifier,X_in,Y_in,X_out,Y_out,preds_in,preds_out,verbose=False,problem_type=0): #problem_type= 0 for classification and 1 for regression
	class_names = ["Worst","Worse","Average","Better","Best"]
	if verbose:
		print("-Training Set Metrics:\n")
		if not problem_type: # Classification metrics
			cnf_matrix = metrics.confusion_matrix(Y_in, preds_in)
			np.set_printoptions(precision=2)
			# Plot non-normalized confusion matrix
			plt.figure()
			plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
			# Plot normalized confusion matrix
			plt.figure()
			plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
			plt.show()

			print("{0}Classification Report(Training set):\n\n".format('-'*30 +'\n'),metrics.classification_report(Y_in,preds_in,target_names=class_names))
		else: # Regression metrics
			print("Mean Absolute Error(MAE):", metrics.mean_absolute_error(Y_in, preds_in))
			print('Mean squared error for training data:', metrics.mean_squared_error(Y_in, preds_in))	
			print("Training set score(sklearn):",classifier.score(X_in,Y_in))
			print("\n")

	print("-Test Set Metrics:\n")
	if not problem_type:
		cnf_matrix = metrics.confusion_matrix(Y_out, preds_out)
		np.set_printoptions(precision=2)
		# Plot non-normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
		# Plot normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
		plt.show()

		print("{0}Classification Report(Test set):\n\n".format('-'*30 +'\n'),metrics.classification_report(Y_out,preds_out,target_names=class_names))
	else:
		print("Mean Absolute Error(MAE):", metrics.mean_absolute_error(Y_out, preds_out))
		print("Mean squared error(MSE):" , metrics.mean_squared_error(Y_out, preds_out))
		print("Test set score(sklearn):" , classifier.score(X_out,Y_out))#score meaning depends on classifier


def classify(datapath,v, normalize=True):#datapath: directory name of the datasets, (v)erbose: True or false, normalize = True normalizes training data
	# Grab both wine datasets in one dataset
	concat_data = get_data(datapath)
	# Bag data to 5 scores 
	recode = {3:0, 4:0, 5:1, 6:2, 7:3, 8:4,9:4}
	concat_data['quality_c'] = bag_data(recode,concat_data,'quality')

	# Split up dataset 70/30 training,testing
	y_wine = concat_data['quality_c']
	X_wine = concat_data.drop(['quality_c','quality'], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3, random_state=420)

	X_train_c , X_test_c = X_train.copy(), X_test.copy() #save test and train X sets for classification

	if normalize:
		# Normalize training examples by removing mean and scaling by interquartile range (better than using s.d=1 for outliers in dataset)
		sclr = RobustScaler()
		X_train = sclr.fit_transform(X_train)
		# Retain Training scale params for scaling test set
		scl_params = sclr.get_params()
		# Normalise test examples using training set normalization params
		sclr = sclr.set_params(**scl_params)
		X_test = sclr.transform(X_test)

	# Set parameters by cross validation
	#==========================================================================================
	# REGRESSION PROBLEM
	#==========================================================================================
	# Multivariate Linear Regression
	clf = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nLinear Regression:\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v,problem_type=1)
	#==========================================================================================
	# Support Vector Machine(kernel=rbf), Regression
	clf = svm.SVR(C=3,kernel='rbf')
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nSVR :\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v,problem_type=1)
	#==========================================================================================
	# NN Regression, default params
	# Grid Search
	h_max = 2 #specify maximum number of hidden layers
	hidden_layer_sizes = build_grid(h_max)
	tuned_param = {'hidden_layer_sizes': hidden_layer_sizes}
	clf = GridSearchCV(neural_network.MLPRegressor(),tuned_param,cv=3) 
	clf.fit(X_train,y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nNNs :\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v,problem_type=1)
	print("Best params:", clf.best_params_)
	#==========================================================================================
	# CLASSIFICATION PROBLEM
	#==========================================================================================
	# Restore normalized examples back to original
	X_train, X_test = X_train_c, X_test_c
	# Support Vector Machine(Kernel=rbf), Classification
	clf = svm.SVC(C=3,kernel='rbf',random_state=0)
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nSVC :\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v)
	#==========================================================================================
	# Support Vector Machine(Kernel=rbf), One vs Rest Classification
	clf = OneVsRestClassifier(estimator=svm.SVC(C=3,kernel='rbf', random_state=1))
	clf.fit(X_train, y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nSVC(OneVsRest):\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v)
	#==========================================================================================
	# NN Classification
	# Grid Search
	h_max = 2 #specify maximum number of hidden layers
	hidden_layer_sizes = build_grid(h_max)
	tuned_param = {'hidden_layer_sizes': hidden_layer_sizes}
	clf = GridSearchCV(neural_network.MLPClassifier(),tuned_param,cv=3)
	clf.fit(X_train,y_train)
	# Make Predictions for both sets
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	print('='*100+"\nNNs :\n")
	print_metrics(clf,X_train,y_train,X_test,y_test,pred_train,pred_test,verbose=v)
	print("Best params:", clf.best_params_)
	#==========================================================================================

def build_grid(h_max=2):
	out = []
	for i in range(10,110,10):
		param = []
		for layer in range(h_max):
			param.append(i)
		param = tuple(param)
		out.append(param)

	return out

if __name__ == '__main__':
	data_path = "datasets"
	verbose = True
	classify(data_path,verbose)
