import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt



def get_data(datdir):
	#get_data  : Grabs entire dataset from given dataset directory name, 'datdir' 
	#oututs tuple of red wine and white wine data
	#Load red and white wine datasets (; separated)
	red_dat = pd.read_csv( "./" + datdir + "/winequality-red.csv",';')
	white_dat = pd.read_csv( "./" + datdir + "/winequality-white.csv", ';')
	out = [red_dat, white_dat]
	out = pd.concat(out)
	return out

def bag_data(map,data,attr):
	#sklearn stub for data recoding(bagging)
	return data[attr].map(map)

def input_training_set(inp):
	#create input functions for tf
	features = {}
	for attr in inp.columns.values:
		features[attr] = inp[attr].as_matrix()
	labels = inp['quality'].as_matrix()
	return features,labels


def one_hot_label(labels):
	#converts target values into one hot encoding
	out = np.array([])
	for val in labels:
		row = np.zeros(9)
		print (val)
		row[val-1] = 1
		out = np.append(out,row,axis=0)
	return out

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    COPYRIGHT SKLEARN : http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

