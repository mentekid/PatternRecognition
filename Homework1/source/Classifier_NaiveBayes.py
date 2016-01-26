from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Load Data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
y = pd.read_csv('data/bugs_train.csv', ';')
X.set_index('classid', inplace=True)
y.set_index('classid', inplace=True)


"""
Gaussian Naive Bayes Classifier
"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

"""
Crossvalidate
"""
import GeneralCrossValidation as gcv
results = gcv.CrossValidateSMOTE(X, y, gnb)



"""
Present Results
"""
[acc, fm, rc, prc] = map(lambda x: np.mean(x), results[0:4])

cm = results[-1]

print "Accuracy:", round(100*acc,2)
print "Precision:", round(100*prc,2)
print "Recall:", round(100*rc,2)
print "F-measure:", round(100*fm,2)
print "Confusion Matrix: ", cm


bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
print pdr