import matplotlib.pyplot as plt
import numpy as np

""" Load data """
import data_read as dtrd
data, features, projects_true = dtrd.All(sparse=False)



plots = [None]*len(range(1,11))

""" Preprocessing """
import mypreprocessing as prp
data = prp.RowWiseNorm(data)

silh = []
comp = []
homo = []
vmea = []
from sklearn.cluster import SpectralClustering
from sklearn import metrics

for k in range(2, 11):
    print k
    km = SpectralClustering(n_clusters=k)
    km.fit(data)
    silh.append(metrics.silhouette_score(data, km.labels_))
    comp.append(metrics.completeness_score(projects_true, km.labels_))
    homo.append(metrics.homogeneity_score(projects_true, km.labels_))
    vmea.append(metrics.v_measure_score(projects_true, km.labels_))

plt.style.use('fivethirtyeight')
plt.plot(range(2,11), silh)
plt.plot(range(2,11), comp)
plt.plot(range(2,11), homo)
plt.plot(range(2,11), vmea)
plt.title('Spectral clustering, Row-wise Normalization')
plt.xlabel('k clusters')
plt.ylabel('Silhouette Coefficient')
plt.legend(['silhouette', 'completeness', 'homogeneity', 'v-measure'], loc='upper right')
plt.show()
