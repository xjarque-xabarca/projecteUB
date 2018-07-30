import pandas as pd
import io
import os.path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import DataSetHandler
from collections import Counter


path_datasets = '/opt/datascience_info/projecteUB/python/'
path_datasets += 'seeds.csv'
#path_datasets += 'seeds_wheat.csv'
#path_datasets += 'HTRU_2.csv'
#path_datasets += '/bank-additional.csv'

df = DataSetHandler.loadCSV( path_datasets )
print( df.head().T )
print( df.dtypes )


#----------------------------------------------------------
#  BARPLOT  CIRCULAR
#----------------------------------------------------------
# print (DataSetHandler.getHistogram2PGN(df))


#----------------------------------------------------------
#  CORRELATION  MATRIX
#----------------------------------------------------------
# DataSetHandler.correlationPGN(df)


#----------------------------------------------------------
#  COMPARE  CLASSIFIERS
#----------------------------------------------------------

strClassifiersss = [ "Linear SVC",  "KNeighbors", "RF-Bagging", "RandomForest", "AdaBoost",  "DecisionTree", "ExtraTrees", "Ridge", "SGD", "LinearDiscriminant", "QuadraticDiscriminant" ]
#strClassifiersss = [ "SVC" ]
#strClassifiersss = [ ]

scores = DataSetHandler.compare_classifiers(df, strClassifiersss)
print( scores )

