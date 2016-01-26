# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:14:12 2016

@author: et3rn1ty
"""
import numpy as np
import matplotlib.pyplot as plt

def frange(x, y, jump): 
    while x < y: 
        yield x 
        x += jump


""" Load data """
import data_read as dtrd
data, features, projects_true = dtrd.All(sparse=False)


""" preprocess """
import mypreprocessing as prp
data = prp.TrimmingPresence(data, low_thresh=4, hig_thresh=70)
data = prp.tfidf(data)
data = np.array(data.todense())


""" DBSCAN algorithm """
from sklearn.cluster import DBSCAN

clus=[]
homo=[]
comp=[]
vmea=[]

choices = [x for x in frange(0.01, 0.2, 0.0001)]

for eps in choices:
    print eps
    db = DBSCAN(eps=eps, min_samples=4, metric='hamming', algorithm='brute').fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    projects = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(projects)) - (1 if -1 in projects else 0)
    
    import sklearn.metrics as metrics
    clus.append( n_clusters_)
    homo.append(metrics.homogeneity_score(projects_true, projects))
    comp.append(metrics.completeness_score(projects_true, projects))
    vmea.append(metrics.v_measure_score(projects_true, projects))

plt.plot(choices, clus)
plt.plot(choices, homo)
plt.plot(choices, comp)
plt.plot(choices, vmea)
plt.legend(['number of clusters', 'homogeneity', 'completenes', 'v-measure'])
plt.xlabel('epsilon distance')
plt.ylabel('score')
plt.title('DBSCAN, tfidf')
plt.show()