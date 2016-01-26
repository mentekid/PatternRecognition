import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GeneralCrossValidation as gcv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""
Load Data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
y = pd.read_csv('data/bugs_train.csv', ';')
X.set_index('classid', inplace=True)
y.set_index('classid', inplace=True)
X = X.as_matrix()
X = X.astype(float)



    
"""
Crossvalidate the pipeline model
"""

accuracy = []
fmeasure = []
recall = []
precision = []
for k in xrange(1, 10, 1):
	cw = {0:1, 1:k}
	pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))
	acc, fm, rc, prc, cm = gcv.CrossValidateSMOTE(X, y, pipe, folds=10)
	accuracy.append(np.mean(acc))
	fmeasure.append(np.mean(fm))
	recall.append(np.mean(rc))
	precision.append(np.mean(prc))
'''
Plot metrics with k as variable
'''

s = plt.plot(xrange(1, 10, 1), accuracy, 'b' , \
    xrange(1, 10, 1), fmeasure, 'r', \
    xrange(1, 10, 1), recall, 'c', \
    xrange(1, 10, 1), precision, 'm')
plt.title('Metrics for different k')
plt.xlabel('k weight value')
plt.ylabel('Metric')
plt.legend(['Accuracy', 'Fmeasure', 'Recall', 'Precision'])
plt.savefig('svm_over_k_weight_metrics.eps', format='eps', dpi=200)
#plt.show(s)

"""
indexes of optimal metrics
"""
opt_fm = fmeasure.index(max(fmeasure))
opt_pr = precision.index(max(precision))


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

#Optimal model according to F-MEASURE
opt = opt_fm +1
cw = {0:1, 1:opt}
pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight=cw))
acc, fm, rc, prc, cm = gcv.CrossValidateSMOTE(X, y, pipe, folds=10)
print cm
#plot confusion matrix
plt.figure(2)
t = plt.imshow(cm)
plt.title('Confusion Matrix for k= '+str(opt)+' and '+str(round(accuracy[opt_fm]*100,2))+'% accuracy')
#plt.title('Confusion Matrix')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues);plt.colorbar()
plt.xticks([0, 1]);plt.yticks([0, 1]);plt.xlabel('Predicted Value');plt.ylabel('True Value')
#plt.show(t)
plt.savefig('svm_conf_matrix.eps', format='eps', dpi=200)




