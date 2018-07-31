import pandas as pd
import io
import os.path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import DataSetHandler
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier



#handler = DataSetHandler()

df = DataSetHandler.loadCSV( '/opt/datascience_info/projecteUB/python/seeds.csv' )
#df = DataSetHandler.loadCSV( '/home/xabarca/Baixades/data_science/projecte/datascience/xabarca-python/HTRU_2.csv' )
#df = DataSetHandler.loadCSV( '/home/xabarca/Baixades/data_science/projecte/datascience/xabarca-python/bank-additional.csv' )
# print( df.head().T )
# colNames = ['area','perimeter','compactness','kernel_length','kernel_width','asymmetry_coeff','kernel_groove_length', 'y']
# df.columns = colNames
# c = Counter(df['y'])
# print( c )

print (DataSetHandler.getHistogram2PGN(df))

# DataSetHandler.correlationPGN(df)

classifiersss = ( [ "LinearSVC()", LinearSVC() ],
         [ "KNeighborsClassifier()", KNeighborsClassifier() ],
         [ "BaggingClassifier()", BaggingClassifier() ],
         [ "RandomForestClassifier()", RandomForestClassifier() ]
     )
scores = DataSetHandler.compare_classifiers(df, classifiersss)
path = DataSetHandler.plot_scores ( scores )
print( path )
# pattttt = DataSetHandler.plot_feature_importances_cancer ( scores )


# target_names = ["2" , "3", "1"]
# plt.rcParams["figure.figsize"] = [10,5]
# plt.pie([c[i] / len(df['area']) * 100.0 for i in c], labels=target_names, 
#         colors=['gold', 'yellowgreen', 'lightcoral'],
#         autopct='%1.1f%%', shadow=True, startangle=90)
# plt.axis('equal')
# plt.show()



#----------------------------------------------------------
#  CHOOSE DATASET OLD STYLE
#----------------------------------------------------------
#manager.loadDataSet("seeds")
#manager.loadDataSet("htru2")
# manager.loadDataSet("bank")
# df = manager.dataset
# print( df.info() )

# sns.pairplot(data=df, kind="reg", hue="y", diag_kind="kde", diag_kws=dict(shade=True))
# plt.show()

#----------------------------------------------------------
# TEST SIMPLE ALGORITHM:  K-NEIGHBORS
#----------------------------------------------------------

# manager.testKNeighbors()
# plt.plot( manager.kn_k, manager.kn_scores )
# plt.title("KNeighbors accuracy")
# plt.xlabel( '# neighbors')
# plt.ylabel( 'Accuracy' )
# plt.show()


#----------------------------------------------------------
# MODEL SELECTION
#----------------------------------------------------------
#manager.modelSelection()
#accuracyData = manager.model_selection_accuracy
#plt.boxplot(accuracyData)
#for i in np.arange(8):
#    xderiv = (i+1)*np.ones(accuracyData[:,i].shape)+(np.random.rand(10,)-0.5)*0.1
#    plt.plot(xderiv,accuracyData[:,i],'ro',alpha=0.3)
#ax = plt.gca()
#ax.set_xticklabels(['Linear','1-NN','3-NN','SVM','DT','bagging', 'RF', 'Adaboost'])
#plt.ylabel('Accuracy')
#plt.show()
#
#
#----------------------------------------------------------
#  PRINCIPAL COMPONENT ANALYSIS  ( P C A )
#----------------------------------------------------------
# manager.pca()

# def myplot(scoreXXXX,coeffKKK,labels=None):
#      xs = score[:,0]
#      ys = score[:,1]
#      n = coeff.shape[0]

#      plt.scatter(xs ,ys, c = y) #without scaling
#      for i in range(n):
#          plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#          if labels is None:
#              plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#          else:
#              plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

#      plt.xlabel("PC{}".format(1))
#      plt.ylabel("PC{}".format(2))
#      plt.grid()

#      #Call the function. 
#      myplot(x_new[:,0:2], pca.components_) 
#      plt.show()
