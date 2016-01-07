from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Load data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
X.set_index('classid', inplace=True)
y = pd.read_csv('data/bugs_train.csv', ';')
y.set_index('classid', inplace=True)


"""
Optimal Estimators: 600
"""
import GeneralCrossValidation as gcv
estimators = 600
class_weights = {0:1, 1:9}
msp = 35

rfc = RandomForestClassifier(n_estimators=estimators, min_samples_split=msp, n_jobs=-1, class_weight=class_weights )
results = gcv.CrossValidateSMOTE(X, y, rfc)
[acc, fm, rc, prc] = map(lambda x: np.mean(x), results[0:4])


cm = results[-1]

print "Accuracy:", round(100*acc,2)
print "Precision:", round(100*prc,2)
print "Recall:", round(100*rc,2)
print "F-measure:", round(100*fm,2)
print "Confusion Matrix: ", cm

bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
print pdr
raw_input()


    
rfc = RandomForestClassifier(n_estimators=estimators, min_samples_split=msp, n_jobs=-1, class_weight=class_weights, criterion='entropy' )
results = gcv.CrossValidateSMOTE(X, y, rfc)
[acc, fm, rc, prc] = map(lambda x: np.mean(x), results[0:4])

cm = results[-1]

print "Accuracy:", round(100*acc,2)
print "Precision:", round(100*prc,2)
print "Recall:", round(100*rc,2)
print "F-measure:", round(100*fm,2)
print "Confusion Matrix: ", cm

bdr = float(cm[1][1])/(cm[1][1]+cm[1][0])
print pdr