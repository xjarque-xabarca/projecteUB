import pandas as pd
from collections import Counter
import os
import matplotlib as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


WORKING_PATH, filename = os.path.split(os.path.abspath("__file__"))


def loadCSV(path):
    df = pd.read_csv(path, sep=";")
    df[ df.columns.values[-1] ] = df[ df.columns.values[-1] ].astype('category')
    return df


def getHistogramPGN(df):
    y = df.iloc[:,-1]
    perc_win = y.sum() / y.count()
    height = [1-perc_win, perc_win]
    bars = ('win=0 (%)', 'win=1 (%)')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("labels histogram")
    path = WORKING_PATH + "/___histogram.png"
    plt.savefig(path) 
    return path


def getHistogram2PGN(df):
    y = df.iloc[:,-1]
    c = Counter(y)
    numDiffClasses = len( y.unique() )
    target_names = y.unique()
    colors = colors=['lightcoral', 'gold', 'yellowgreen', 'cyan'][:numDiffClasses]
    plt.rcParams["figure.figsize"] = [10,5]
    plt.pie([c[i] / len(y) * 100.0 for i in c], 
            labels=target_names, 
            colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title( df.columns.values[-1] )
    path = WORKING_PATH + "/___circle_labels.png"
    plt.savefig(path)
    plt.clf()
    return path


def correlationPGN(dataset):
    allColsAsFeatures = dataset.columns.values[:-1]
    X = dataset[ allColsAsFeatures ]
    corr_matrix = X.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns_corrplot = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    sns_corrplot.get_figure()
    path = WORKING_PATH + "/___corrrr.png"
    plt.savefig(path)
    plt.clf()
    return path


def dictionayClassifiers(keyClassifier):
    if (keyClassifier=="SVC"):
        return [ keyClassifier, SVC(kernel="rbf", C=0.025, probability=True) ]
    if (keyClassifier=="Linear SVC"):
        return [ keyClassifier, LinearSVC() ]
    if (keyClassifier=="KNeighbors"):
        return [ keyClassifier, KNeighborsClassifier(3) ]
    if (keyClassifier=="RF-Bagging"):
        return  [ keyClassifier, BaggingClassifier() ]
    if (keyClassifier=="RandomForest"):
        return [ keyClassifier, RandomForestClassifier() ]
    if (keyClassifier=="AdaBoost"):
        return [ keyClassifier, AdaBoostClassifier() ]
    if (keyClassifier=="DecisionTree"):
        return [ keyClassifier, DecisionTreeClassifier() ]
    if (keyClassifier=="ExtraTrees"):
        return [ keyClassifier, ExtraTreesClassifier() ]
    if (keyClassifier=="Ridge"):
        return [ keyClassifier, RidgeClassifier() ]
    if (keyClassifier=="SGD"):
        return [ keyClassifier, SGDClassifier() ]
    if (keyClassifier=="GaussianNB"):
        return [ keyClassifier, GaussianNB() ]
    if (keyClassifier=="LinearDiscriminant"):
        return [ keyClassifier, LinearDiscriminantAnalysis() ]
    if (keyClassifier=="QuadraticDiscriminant"):
        return [ keyClassifier, QuadraticDiscriminantAnalysis() ]
    return None


def compare_classifiers(dataset, strClassifiers):
	classifiers = []
	for strClass in strClassifiers:
		clsf = dictionayClassifiers(strClass)
		if clsf is not None:
			classifiers.append( clsf )
    
	if ( len(classifiers)>0 ):
		allColsAsFeatures = dataset.columns.values[:-1]
		X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
		
		# a bit of wrangling...
		X = pd.get_dummies(X)
		X.dropna(inplace=True)
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

		# given list of models do a benchmark
		unsorted_scores = []
		
		for name, model in classifiers:
			# print('Model {}'.format(name))
			scores = cross_val_score(model, X_train, y_train, cv=StratifiedShuffleSplit(n_splits=5, random_state=0))
			# predicting for the test set
			test_score = accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test)
			score_tupple = [ name, np.mean(scores), np.std(scores), np.min(scores), np.max(scores), test_score ]
			unsorted_scores.append(score_tupple)
		
		scores = sorted(unsorted_scores, key=lambda x: -x[1])
		df = pd.DataFrame( scores )
		df.columns = ['Model', 'Mean Val. Accuracy', 'Std Val. Accuracy', 'min', 'max', 'Test set Accuracy']
		
	else:
		df = pd.DataFrame()

	return df


def plot_scores(scores):
    plt.figure(figsize=(15, 6))
    names, val_scores = [name for name,_,_,_,_,_,_ in scores], [score for _, score,_,_,_,_,_ in scores]
    ax = sns.barplot(x=names, y=val_scores)
    
    for p, score in zip(ax.patches, val_scores):
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 0.005,
                '{:1.3f}'.format(score),
                ha="center", fontsize=14) 
        
    plt.xlabel('method', fontsize=18)
    plt.ylabel('Mean Val. Accuracy', fontsize=18)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0.6, 1)
    path = WORKING_PATH + "/___comparison.png"
    plt.savefig(path)
    plt.clf()
    return path

