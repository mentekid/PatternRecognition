import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import GeneralCrossValidation as gcv
import matplotlib.pyplot as plt

"""
Load data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
X.set_index('classid', inplace=True)
y = pd.read_csv('data/bugs_train.csv', ';')
y.set_index('classid', inplace=True)



"""
Grid search for k
"""
accuracy = []
fmeasure = []
recall = []
precision = []
for k in xrange(1, 63, 1):
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance', metric='chebyshev')
    sc, fm, rc, prc, cm = gcv.CrossValidateSMOTE(X, y, knn, folds=10)
    accuracy.append(np.mean(sc))
    fmeasure.append(np.mean(fm))
    recall.append(np.mean(rc))
    precision.append(np.mean(prc))

'''
Plot metrics with k as variable
'''

s = plt.plot(xrange(1, 63, 1), accuracy, 'b' , \
    xrange(1, 63, 1), fmeasure, 'r', \
    xrange(1, 63, 1), recall, 'c', \
    xrange(1, 63, 1), precision, 'm')
plt.title('Metrics for different k')
plt.xlabel('k value')
plt.ylabel('Metric')
plt.legend(['Accuracy', 'Fmeasure', 'Recall', 'Precision'])
#plt.savefig('knn_over_k_metrics.eps', format='eps', dpi=200)
plt.show(s)

"""
indexes of optimal metrics
"""

opt=accuracy.index(max(accuracy))
opt_re = recall.index(max(recall))
opt_fm = fmeasure.index(max(fmeasure))
opt_pr = precision.index(max(precision))

print 'OPTIMAL ACCURACY'
print 'accuracy k(acc)= ', opt , 'accuracy value= ', accuracy[opt]
print 'recall k(acc)= ', opt, 'recall value= ', recall[opt]
print 'fmeasure k(acc)= ', opt, 'fmeasure value= ', fmeasure[opt] 
print 'precision k(rec)= ', opt, 'precision value= ', precision[opt]

print 'OPTIMAL RECALL'
print 'accuracy k(rec)= ', opt_re , 'accuracy value= ', accuracy[opt_re]
print 'recall k(rec)= ', opt_re, 'recall value= ', recall[opt_re]
print 'fmeasure k(rec)= ', opt_re, 'fmeasure value= ', fmeasure[opt_re] 
print 'precision k(rec)= ', opt_re, 'precision value= ', precision[opt_re]

print 'OPTIMAL FMEASURE'
print 'accuracy k(fm)= ', opt_fm , 'accuracy value= ', accuracy[opt_fm]
print 'recall k(fm)= ', opt_fm, 'recall value= ', recall[opt_fm]
print 'fmeasure k(fm)= ', opt_fm, 'fmeasure value= ', fmeasure[opt_fm]
print 'precision k(rec)= ', opt_fm, 'precision value= ', precision[opt_fm]

print 'OPTIMAL PRECISION'
print 'accuracy k(fm)= ', opt_pr , 'accuracy value= ', accuracy[opt_pr]
print 'recall k(fm)= ', opt_pr, 'recall value= ', recall[opt_pr]
print 'fmeasure k(fm)= ', opt_pr, 'fmeasure value= ', fmeasure[opt_pr]
print 'precision k(rec)= ', opt_pr, 'precision value= ', precision[opt_pr]
"""
Plot confusion matrix with optimal recall
"""

k_opt = opt_fm # index of optimal PRECISION
knn = KNeighborsClassifier(n_neighbors=k_opt ,weights='distance', metric='chebyshev')
acc, fm, rc, prc, cm = gcv.CrossValidateSMOTE(X, y, knn, folds=10)
print cm

plt.figure(2)
t = plt.imshow(cm)
plt.title('Confusion Matrix for k= '+str(k_opt)+' and '+str(round(accuracy[k_opt]*100,2))+'% accuracy')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues);plt.colorbar()
plt.xticks([0, 1]);plt.yticks([0, 1]);plt.xlabel('Predicted Value');plt.ylabel('True Value')
plt.show(t)
#plt.savefig('knn_conf_matrix.eps', format='eps', dpi=200)
#OPTIMAL RECALL


