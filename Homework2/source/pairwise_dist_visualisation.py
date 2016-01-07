# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:29:08 2016

@author: et3rn1ty
"""
import numpy as np
import data_read as dtrd
data, features, projects_true = dtrd.All(sparse=False)

""" Processing """
import mypreprocessing as prp

data = prp.TrimmingPresence(data, low_thresh=1, hig_thresh=70)
data = prp.tfidf(data)
data = np.array(data.todense())

from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
distances = ['cosine', 'canberra', 'chebyshev','hamming',  'minkowski', 'euclidean']

for i, dist in enumerate(distances):
    D = pairwise_distances(data, metric=dist)
    #plots[i] = plt.figure(i)
    try:
        sil = silhouette_score(data, np.array(projects_true), metric=dist)
    except ValueError:
        sil = -2
    plt.imshow(D, interpolation='nearest')
    plt.title(str(dist+' pairwise distances, silhouette coeff='+str(sil)))
    plt.colorbar()
    plt.show()
    
    if i%5==0:
        raw_input('Press Enter to continue')
    
raw_input()
    