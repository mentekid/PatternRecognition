import MetaClassifier_Class as mclf
mclf = reload(mclf)
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib.pyplot as plt

def shuffle_in_unison_inplace(a, b):
    shuf = np.random.permutation(a.index)
    a=a.reindex(shuf)
    b=b.reindex(shuf)
    return (a,b)

"""
Load Data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
y = pd.read_csv('data/bugs_train.csv', ';')
X.set_index('classid', inplace=True)
y.set_index('classid', inplace=True)


"""
Create ensemble of classifiers
"""
#unbiased classifiers
cw = {0:1, 1:1}
knn1 = KNeighborsClassifier(n_neighbors=12, metric='chebyshev')
knn2 = KNeighborsClassifier(n_neighbors=9, metric='canberra')
giniTree_unbiased = tree.DecisionTreeClassifier(criterion='gini', class_weight=cw)
entropyTree_unbiased = tree.DecisionTreeClassifier(criterion='entropy', class_weight=cw)
giniRfc_unbiased = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw )
entropyRfc_unbiased = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw, criterion='entropy')
svm_unbiased = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))
gnb = GaussianNB()

unbiased = [knn1, knn2, giniTree_unbiased, entropyTree_unbiased, giniRfc_unbiased, entropyRfc_unbiased,\
svm_unbiased]

#positively - biased classifiers
cw = {0:1, 1:15}
giniTree = tree.DecisionTreeClassifier(criterion='gini', class_weight=cw, min_samples_split=30)
entropyTree = tree.DecisionTreeClassifier(criterion='entropy', class_weight=cw, min_samples_split=30)
giniRfc = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw )
entropyRfc = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=-1, class_weight=cw, criterion='entropy')
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))
biased = [giniTree, entropyTree, svm]

ensemble = unbiased + biased

"""
Create aggregators
"""
aggregator1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:1}, C=0.5))
aggregator2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:2}, C=0.5))
aggregator3 = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:4}, C=0.5))
#aggregator3 = KNeighborsClassifier(n_neighbors=5, metric='hamming')
aggregator4 = GaussianNB()
aggregator5 = tree.DecisionTreeClassifier(criterion='gini')

aggregators = [aggregator1, aggregator2, aggregator3, aggregator4, aggregator5] #best: slightly biased logistic regressor

"""
Create metaclassifier, train and test (or cross-validate)
"""

accuracy_peragg=[]
bugs_peragg=[]


for aggregator in aggregators:
    metaclassifier = mclf.MetaClassifier(ensemble, aggregator, useSMOTE=True)
    print "Classifier", aggregator
    tries = 5
    accuracy = []
    bugs=[]
    
    for i in range(tries):
        
        X, y = shuffle_in_unison_inplace(X, y)
        print "Training %d of %d" %(i+1, tries)
        #cross-validation
        from GeneralCrossValidation import CrossValidateSMOTE
        
        acc, fmeasure, recall, precision, cm = \
            CrossValidateSMOTE(X, y, metaclassifier, runSMOTE=False)
        
        print np.mean(acc), np.mean(precision), np.mean(recall), np.mean(fmeasure)
        
        accuracy.extend(acc)
        bugs.append(np.mean(precision))
    accuracy_peragg.append(accuracy)
    bugs_peragg.append(bugs)

    
        
                
plt.figure(1)
p1 = plt.boxplot(accuracy_peragg)
plt.xticks([1, 2, 3, 4, 5], ['LR 1:1','LR 1:2', 'LR 1:4', 'Gaussian', 'Tree'])
plt.title('Accuracy of meta-models')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.axis([0, 6, 0, 1])
plt.show(p1)

plt.figure(2)
p2 = plt.boxplot(bugs_peragg)
plt.xticks([1, 2, 3], ['LR','LR_biased 1:2', 'LR_biased 1:4'])
plt.title('Bug Discovery Rate of meta-models')
plt.ylabel('Bug Discovery Rate')
plt.xlabel('Models')
plt.axis([0, 4, 0, 1])
plt.show(p2)