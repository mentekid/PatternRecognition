# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5

@author: Yannis Mentekidis
"""

""" Read data """
import data_read as dtrd
import numpy as np
data, features, projects_true = dtrd.All(sparse=False)

""" Processing """
import mypreprocessing as prp

data = prp.TrimmingPresence(data, low_thresh=1, hig_thresh=70)
#data = prp.LDAProjection(data, features=features, LDA_topics=12,verbose=True)
#data = prp.RowWiseNorm(data)

data = prp.tfidf(data)
data = np.array(data.todense())



""" Clustering """
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

dist_metric = "cosine"
silh = []
homo = []
comp = []
vmea = []
choices = range(2, 11)
for c in choices:
    clu = AgglomerativeClustering(n_clusters = c, affinity=dist_metric, linkage="average")
    clu.fit(data)
    print c
    silh.append(metrics.silhouette_score(data, clu.labels_, metric=dist_metric))
    homo.append(metrics.homogeneity_score(projects_true, clu.labels_))
    comp.append(metrics.completeness_score(projects_true, clu.labels_))
    vmea.append(metrics.v_measure_score(projects_true, clu.labels_))


import matplotlib.pyplot as plt
plt.plot(choices, silh)
plt.plot(choices, homo)
plt.plot(choices, comp)
plt.plot(choices, vmea)
plt.legend(['silhouette', 'homogeneity', 'completeness', 'v-measure'], loc='upper left')
plt.title('Average Linkage Hierarchical Clustering,\n Cosine Distance, TFIDF')
plt.xlabel('Number of Clusters')
plt.ylabel('Metric')
plt.show()