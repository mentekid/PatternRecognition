from __future__ import division
from sklearn import tree #decision tree
import pandas as pd #dataframes, csvread, etc
import numpy as np #matlab-like arrays
import matplotlib.pyplot as plt

import GeneralCrossValidation as gcv

"""
Load data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
X.set_index('classid', inplace=True)
y = pd.read_csv('data/bugs_train.csv', ';')
y.set_index('classid', inplace=True)

accuracy = []
bdrall = []
x = 4
for s in range(2,40, 4):
    cw = {0:1, 1:x}
    giniTree = tree.DecisionTreeClassifier(min_samples_split=s, criterion='entropy', class_weight=cw)
    results = gcv.CrossValidateSMOTE(X, y, giniTree)


    [acc, fm, rc, prc] = map(lambda x: np.mean(x), results[0:4])
    cm = results[-1]
    bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
    accuracy.append(acc)
    bdrall.append(bdr)
    
plt.plot(range(2,40, 4), accuracy, 'b', range(2, 40, 4), bdrall, 'r')
plt.xlabel('Samples Split')
plt.ylabel('Metric score')
plt.title('Decision Tree Classifier - Entropy criterion')
plt.legend(['accuracy', 'bug prediction rate'])
plt.show()

print "Accuracy:", round(100*acc,2)
print "Precision:", round(100*prc,2)
print "Recall:", round(100*rc,2)
print "F-measure:", round(100*fm,2)
print "Confusion Matrix: ", cm
bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
print bdr
raw_input()

entropyTree = tree.DecisionTreeClassifier(min_samples_split=35, criterion='entropy')
results = gcv.CrossValidateSMOTE(X, y, entropyTree)


[acc, fm, rc, prc] = map(lambda x: np.mean(x), results[0:4])
cm = results[-1]

print "Accuracy:", round(100*acc,2)
print "Precision:", round(100*prc,2)
print "Recall:", round(100*rc,2)
print "F-measure:", round(100*fm,2)
print "Confusion Matrix: ", cm

bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
print pdr