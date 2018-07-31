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
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit


WORKING_PATH, filename = os.path.split(os.path.abspath("__file__"))


def loadCSV(path):
    df = pd.read_csv(path, sep=";")
    
    
    
    #df.columns.values[-1]
    
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
    if (keyClassifier=="Linear SVC"):
        return [ keyClassifier, LinearSVC() ]
    if (keyClassifier=="KNeighbors"):
        return [ keyClassifier, KNeighborsClassifier() ]
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
    return None


def compare_classifiers(dataset, strClassifiers):

	classifiers = []
	for strClass in strClassifiers:
		clsf = dictionayClassifiers(strClass)
		if clsf is not None:
			classifiers.append( clsf )
    
	if ( len(classifiers)>0 ):

		# classifiers = ( [ "LinearSVC()", LinearSVC() ],
		#      [ "KNeighborsClassifier()", KNeighborsClassifier() ],
		#      [ "BaggingClassifier()", BaggingClassifier() ],
		#      [ "RandomForestClassifier()", RandomForestClassifier() ]
		# )

		allColsAsFeatures = dataset.columns.values[:-1]
		X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

		# given list of models do a benchmark
		unsorted_scores = []
		
		for name, model in classifiers:
			#print('Model {}'.format(name))
			scores = cross_val_score(model, X_train, y_train, cv=StratifiedShuffleSplit(n_splits=5, random_state=0))
			#scores = cross_val_score(model, X_train, y_train, cv=5)
			#repdicting for the test set
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


def plot_feature_importances_cancer(scores):
    names, val_scores = [name for name,_,_,_,_,_,_ in scores], [score for _, score,_,_,_,_,_ in scores]

    plt.rcParams["figure.figsize"] = [15,9]
    n_features = len(names)
    plt.barh(range(n_features), val_scores, align='center')
    plt.yticks(np.arange(n_features), names)
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    path = WORKING_PATH + "/___comparison_cancer_model.png"
    plt.savefig(path) 
    plt.clf()
    return path



# def compare_classifiers(X_train_imp, y_train, X_test_imp, y_test, classifiers, cv=5, 
#                         test_size=0.2):
#     # given list of models do a benchmark
#     unsorted_scores = []
    
#     for name, model in classifiers:
#         #print('Model {}'.format(name))
#         scores = cross_val_score(model, X_train_imp, y_train, cv=StratifiedShuffleSplit(n_splits=5, random_state=0))
#         #repdicting for the test set
#         test_score = accuracy_score(model.fit(X_train_imp, y_train).predict(X_test_imp), y_test)
#         score_tupple = (name, np.mean(scores), np.std(scores), np.min(scores), np.max(scores), test_score, scores)
#         unsorted_scores.append(score_tupple)

#     scores = sorted(unsorted_scores, key=lambda x: -x[1])
#     print (tabulate([x[:-1] for x in scores], floatfmt=".3f", 
#                     headers=('model', 'Mean Val. Accuracy', 'Std Val. Accuracy', 'min', 'max', 'Test set Accuracy')))
    
#     return scores


# def print_scores(unsorted_scores):
#     scores = sorted(unsorted_scores, key=lambda x: -x[1])
#     print (tabulate([x[:-1] for x in scores], floatfmt=".3f", 
#                     headers=("model", 'Mean Val. Accuracy', 'Std Val. Accuracy', 'min', 'max', 'Test set Accuracy')))

    
# def plot_scores(scores):
#     plt.figure(figsize=(15, 6))
#     names, val_scores = [name for name,_,_,_,_,_,_ in scores], [score for _, score,_,_,_,_,_ in scores]
#     ax = sns.barplot(x=names, y=val_scores)
    
#     for p, score in zip(ax.patches, val_scores):
#         height = p.get_height()
#         ax.text(p.get_x()+p.get_width()/2.,
#                 height + 0.005,
#                 '{:1.3f}'.format(score),
#                 ha="center", fontsize=14) 
        
#     plt.xlabel('method', fontsize=18)
#     plt.ylabel('Mean Val. Accuracy', fontsize=18)
#     plt.xticks(rotation=90, fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.show()  






















# def loadDataSet(strDataset):
#     df = None
#     if ( strDataset == "seeds"):
#         df = getdataSeeds(WORKING_PATH)
#     if ( strDataset == "bank"):
#         df = getdataBank(WORKING_PATH)
#     if ( strDataset == "htru2"):
#         df = getdataHtru2(WORKING_PATH)
#     return df


#def getLabel(dataset):
#    return dataset.iloc[:,-1]
#
#
#def getDataSetInfo(dataset):
#    return dataset.info()   
#
#
#def getDataSetHead(dataset):
#    return dataset.head().T
#    
#
#def testKNeighbors(dataset):
#    allColsAsFeatures = dataset.columns.values[:-1]
#    X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
#    #print( "X shape = " , X.shape, " ,  y shape = " , y.shape )
#
#    PRC = 0.4
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC, random_state=666)
#
#    scores = []
#    k_values = np.arange(1, 11)
#    for k in k_values:
#        classifier = KNeighborsClassifier(n_neighbors=k )
#        classifier.fit(X_train, y_train)
#        #yhat = classifier.predict(X_test)
#        #matrix_confusion = metrics.confusion_matrix( yhat, y_test )
#        #print( k, "matrix_confusion" , matrix_confusion )
#        scores.append( classifier.score(X_test, y_test) )
#
#    # self.kn_k = k_values
#    # self.kn_scores = scores
#
#    return
#
#
#def modelSelection_backup(dataset):
#    allColsAsFeatures = dataset.columns.values[:-1]
#    X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
#    PRC = 0.2
#    seed = 7
#    num_trees = 30
#    acc_r=np.zeros((10,7))
#    for i in np.arange(10):
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
#        nn1 = KNeighborsClassifier(n_neighbors=1)
#        nn3 = KNeighborsClassifier(n_neighbors=3)
#        svc = svm.SVC()
#        dt = tree.DecisionTreeClassifier()
#        bagg = BaggingClassifier(base_estimator=None)
#        rf = RandomForestClassifier()
#        ada = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
#        
#        nn1.fit(X_train,y_train)
#        nn3.fit(X_train,y_train)
#        svc.fit(X_train,y_train)
#        dt.fit(X_train,y_train)
#        bagg.fit(X_train,y_train)
#        rf.fit(X_train,y_train)
#        ada.fit(X_train,y_train)
#        
#        yhat_nn1=nn1.predict(X_test)
#        yhat_nn3=nn3.predict(X_test)
#        yhat_svc=svc.predict(X_test)
#        yhat_dt=dt.predict(X_test)
#        yhat_bagg=bagg.predict(X_test)
#        yhat_rf=rf.predict(X_test)
#        yhat_ada=ada.predict(X_test)
#        
#        acc_r[i][0] = metrics.accuracy_score(yhat_nn1, y_test)
#        acc_r[i][1] = metrics.accuracy_score(yhat_nn3, y_test)
#        acc_r[i][2] = metrics.accuracy_score(yhat_svc, y_test)
#        acc_r[i][3] = metrics.accuracy_score(yhat_dt, y_test)
#        acc_r[i][4] = metrics.accuracy_score(yhat_bagg, y_test)
#        acc_r[i][5] = metrics.accuracy_score(yhat_rf, y_test)
#        acc_r[i][6] = metrics.accuracy_score(yhat_ada, y_test)
#
#    return acc_r
#
#    # plt.boxplot(acc_r)
#    # for i in np.arange(7):
#    #     xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
#    #     plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
#
#
#
#def modelSelection(dataset):
#    allColsAsFeatures = dataset.columns.values[:-1]
#    X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
#    PRC = 0.2
#    seed = 7
#    num_trees = 30
#    acc_r=np.zeros((10,8))
#    for i in np.arange(10):
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PRC)
#        lreg = LogisticRegression()
#        nn1 = KNeighborsClassifier(n_neighbors=1)
#        nn3 = KNeighborsClassifier(n_neighbors=3)
#        svc = svm.SVC()
#        dt = tree.DecisionTreeClassifier()
#        bagg = BaggingClassifier(base_estimator=None)
#        rf = RandomForestClassifier()
#        ada = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
#        
#        lreg.fit(X_train,y_train)
#        nn1.fit(X_train,y_train)
#        nn3.fit(X_train,y_train)
#        svc.fit(X_train,y_train)
#        dt.fit(X_train,y_train)
#        bagg.fit(X_train,y_train)
#        rf.fit(X_train,y_train)
#        ada.fit(X_train,y_train)
#        
#        yhat_lreg=lreg.predict(X_test)
#        yhat_nn1=nn1.predict(X_test)
#        yhat_nn3=nn3.predict(X_test)
#        yhat_svc=svc.predict(X_test)
#        yhat_dt=dt.predict(X_test)
#        yhat_bagg=bagg.predict(X_test)
#        yhat_rf=rf.predict(X_test)
#        yhat_ada=ada.predict(X_test)
#        
#        acc_r[i][0] = metrics.accuracy_score(yhat_lreg, y_test)
#        acc_r[i][1] = metrics.accuracy_score(yhat_nn1, y_test)
#        acc_r[i][2] = metrics.accuracy_score(yhat_nn3, y_test)
#        acc_r[i][3] = metrics.accuracy_score(yhat_svc, y_test)
#        acc_r[i][4] = metrics.accuracy_score(yhat_dt, y_test)
#        acc_r[i][5] = metrics.accuracy_score(yhat_bagg, y_test)
#        acc_r[i][6] = metrics.accuracy_score(yhat_rf, y_test)
#        acc_r[i][7] = metrics.accuracy_score(yhat_ada, y_test)
#
#    return acc_r
#
#    # plt.boxplot(acc_r)
#    # for i in np.arange(7):
#    #     xderiv = (i+1)*np.ones(acc_r[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
#    #     plt.plot(xderiv,acc_r[:,i],'ro',alpha=0.3)
#
#
#
#
#def pca(dataset):
#    import matplotlib.pyplot as plt
#
#    allColsAsFeatures = dataset.columns.values[:-1]
#    X,y = dataset[ allColsAsFeatures ], dataset.iloc[:,-1]
#
#    # In general it is a good idea to scale the data
#    scaler = StandardScaler()
#    scaler.fit(X)
#    X=scaler.transform(X)
#
#    pca = PCA()
#    pca.fit(X,y)
#    x_new = pca.transform(X)   
#
#    #Call the function. 
#    myplotPCA(x_new[:,0:2], pca.components_, y) 
#
#    #print( pca.components_ )
#    #print( pca.explained_variance_ratio_ )
#    return
#
#
#
#
#
#def myplotPCA(score,coeff,y, labels=None ):
#	xs = score[:,0]
#	ys = score[:,1]
#	n = coeff.shape[0]
#	y_categorical = y.astype('category')
#
#	plt.scatter(xs ,ys, c=y_categorical.cat.codes) #without scaling
#	for i in range(n):
#		plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#		if labels is None:
#			plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#		else:
#			plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
#
#	plt.xlabel("PC{}".format(1))
#	plt.ylabel("PC{}".format(2))
#	plt.grid()
#	plt.show()
#
#
#
#
#
