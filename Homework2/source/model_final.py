# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:43:07 2016

@author: Yannis Mentekidis, Themis Papavasileiou, Panos Siatos
"""
""" Load Data """
import data_read as dtrd
data, features, projects_true = dtrd.All(sparse=False)

""" Process Data """
import mypreprocessing as prp
import numpy as np
data = prp.TrimmingPresence(data, low_thresh=1, hig_thresh=70)
data_p = prp.tfidf(data)
data_p = np.array(data_p.todense())

""" Clustering with Hierarchical Algorithm """
from sklearn.cluster import AgglomerativeClustering
c=7
dist_metric='cosine'
clu = AgglomerativeClustering(n_clusters = c, affinity=dist_metric, linkage="average")
clu.fit(data_p)

for cluster in range(c):
    print "-=-=-=-=-= Cluster %d -=-=-=-=-=" %(cluster)
    
    indices = [i for i, x in enumerate(list(clu.labels_)) if x == cluster]
    
    print indices