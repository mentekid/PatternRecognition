# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:12:32 2016

@author: Yannis Mentekidis

Tests different aggregators' accuracy and precision for the same ensemble
"""


""" Load Libraries """
# pandas and numpy
import pandas as pd
import numpy as np

#preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#classifiers
from MajorityVoting import MajorityVoter as Voter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import MetaClassifier_Class as mclf

#plotting
import matplotlib.pyplot as plt


""" Define shuffling function """
def shuffle_in_unison_inplace(a, b):
    shuf = np.random.permutation(a.index)
    a=a.reindex(shuf)
    b=b.reindex(shuf)
    return (a,b)



""" Load Data """
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
y = pd.read_csv('data/bugs_train.csv', ';')
X.set_index('classid', inplace=True)
y.set_index('classid', inplace=True)

s
""" Create an ensemble """
ensemble = list()

#unbiased classifiers
knn1 = KNeighborsClassifier(n_neighbors=12, metric='chebyshev')
knn2 = KNeighborsClassifier(n_neighbors=9, metric='canberra')
giniTree_unbiased = tree.DecisionTreeClassifier(criterion='gini')
entropyTree_unbiased = tree.DecisionTreeClassifier(criterion='entropy')
svm_unbiased = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

ensemble.extend(\
[knn1, knn2, giniTree_unbiased, entropyTree_unbiased, svm_unbiased])

#biased classifiers
cw = {0:1, 1:10}
giniTree = tree.DecisionTreeClassifier(criterion='gini', class_weight=cw, min_samples_split=30)
entropyTree = tree.DecisionTreeClassifier(criterion='entropy', class_weight=cw, min_samples_split=30)
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))

ensemble.extend([giniTree, entropyTree, svm])

""" Different aggregator candidates """
agg1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:2}, C=0.5))
agg2 = GaussianNB()
agg3 = tree.DecisionTreeClassifier(criterion='gini')
agg4 = KNeighborsClassifier(n_neighbors=12, metric='hamming')
agg5 = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
agg6 = Voter()

aggregators = [agg1, agg2, agg3, agg4, agg5, agg6]
agg_names = ['Logistic Regression 1:2', 'Gaussian NB', 'Decision Tree', 'kNN', 'SVM', 'voting']

"""
Create metaclassifier, train and test (or cross-validate)
"""

accuracy_peragg=[]
prec_peragg=[]

for aggregator in aggregators:
    metaclassifier = mclf.MetaClassifier(ensemble, aggregator, useSMOTE=True)
    print "Classifier", aggregator
    tries = 10
    accuracy = []
    prec=[]
    
    for i in range(tries):
        
        X, y = shuffle_in_unison_inplace(X, y)
        print "Training %d of %d" %(i+1, tries)
        #cross-validation
        from GeneralCrossValidation import CrossValidateSMOTE
        
        acc, fmeasure, recall, precision, cm = \
            CrossValidateSMOTE(X, y, metaclassifier, runSMOTE=False)
        
        print np.mean(acc), np.mean(precision), np.mean(recall), np.mean(fmeasure)
        
        accuracy.extend(acc)
        prec.extend(precision)
        print prec
    accuracy_peragg.append(accuracy)
    prec_peragg.append(prec)

    
        
             
plt.figure(1)
plt.boxplot(accuracy_peragg)
plt.xticks(range(1, len(agg_names)+1), agg_names)
plt.title('Accuracy of meta-models')
plt.ylabel('Accuracy')
plt.xlabel('Aggregator')
plt.axis([0, len(agg_names)+1, 0, 1])

plt.figure(2)
plt.boxplot(prec_peragg)
plt.xticks(range(1, len(agg_names)+1), agg_names)
plt.title('Precision of meta-models')
plt.ylabel('Precision')
plt.xlabel('Aggregator')
plt.axis([0, len(agg_names)+1, 0, 1])
plt.show()