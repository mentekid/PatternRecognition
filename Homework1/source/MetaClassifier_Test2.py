# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:12:32 2016

@author: Yannis Mentekidis

Tests an aggregator's accuracy and precision for different ensembles
"""


""" Load Libraries """
# pandas and numpy
import pandas as pd
import numpy as np

#preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import MetaClassifier_Class as mclf

#plotting
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


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


""" Create ensembles """

#unbiased classifiers
knn1 = KNeighborsClassifier(n_neighbors=12, metric='chebyshev')
knn2 = KNeighborsClassifier(n_neighbors=9, metric='canberra')
giniTree_unbiased = tree.DecisionTreeClassifier(criterion='gini')
entropyTree_unbiased = tree.DecisionTreeClassifier(criterion='entropy')
svm_unbiased = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
gnb = GaussianNB()
giniRfc = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1)
entropyRfc = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, criterion='entropy')
log_reg_unb = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:1}, C=0.5))

#biased classifiers
cw = {0:1, 1:10}
giniTree = tree.DecisionTreeClassifier(criterion='gini', class_weight=cw, min_samples_split=30)
entropyTree = tree.DecisionTreeClassifier(criterion='entropy', class_weight=cw, min_samples_split=30)
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))
giniRfc_bias = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw )
entropyRfc_bias = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw, criterion='entropy')
log_reg_bias = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:4}, C=0.5))


ens_names = ['big', 'small','good+bad', 'only bad', 'no biased']
ens1 = [knn1, knn2, giniTree_unbiased, entropyTree_unbiased, svm_unbiased,giniRfc, entropyRfc, giniTree, entropyTree, svm, giniRfc_bias, entropyRfc_bias]
ens2 = [knn1, entropyTree, svm, entropyRfc_bias]
ens3 =  ens2+[gnb, log_reg_unb, log_reg_bias]
ens4 = [gnb, log_reg_unb, log_reg_bias, giniTree]
ens5 = [knn1, knn2, giniTree_unbiased, entropyTree_unbiased, svm_unbiased,giniRfc, entropyRfc, gnb, log_reg_unb]
 
ensembles = [ens1, ens2, ens3, ens4, ens5]

""" Best aggregator """
agg1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:2}, C=0.5))


"""
Create metaclassifier, train and test (or cross-validate)
"""

accuracy_peragg=[]
prec_peragg=[]

for (ens, ens_name) in zip(ensembles, ens_names):
    metaclassifier = mclf.MetaClassifier(ens, agg1, useSMOTE=True)
    print "Ensemble",ens_name
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
plt.xticks(range(1, len(ens_names)+1), ens_names)
plt.title('Accuracy of meta-models')
plt.ylabel('Accuracy')
plt.xlabel('Aggregator')
plt.axis([0, len(ens_names)+1, 0, 1])

plt.figure(2)
plt.boxplot(prec_peragg)
plt.xticks(range(1, len(ens_names)+1), ens_names)
plt.title('Precision of meta-models')
plt.ylabel('Precision')
plt.xlabel('Aggregator')
plt.axis([0, len(ens_names)+1, 0, 1])
plt.show()