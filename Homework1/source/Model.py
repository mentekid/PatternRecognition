import MetaClassifier_Class as mclf
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib.pyplot as plt


"""
Load Data
"""
X = pd.read_csv('data/source-code-metrics_train.csv', ';')
y = pd.read_csv('data/bugs_train.csv', ';')
X.set_index('classid', inplace=True)
y.set_index('classid', inplace=True)

X_test = pd.read_csv('data/source-code-metrics_test.csv', ';')
X_test.set_index('classid', inplace=True)


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
Create aggregator
"""
aggregator = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0:1, 1:4}, C=0.5))


"""
Create, train, and predict metaclassifier
""", 
clf.fit(X.as_matrix(), y.as_matrix().ravel())
y_test = clf.predict(X_test.as_matrix())
y_test = pd.DataFrame(y_test, columns=['bugs']) #transform to pandas dataframe