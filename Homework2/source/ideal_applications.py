# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:45:14 2016

@author: et3rn1ty
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

""" k-Means Ideal Problem """
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=3.0 )

# Incorrect number of clusters
y_pred = KMeans(n_clusters=3).fit_predict(X)

plt.figure(1)
plt.style.use('fivethirtyeight')
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-Means Clustering Ideal Geometry")

sil = silhouette_score(X, y_pred)
print sil

""" DBSCAN Ideal Problem """
n_samples = 2000
random_state = 170
X, y = make_moons(n_samples=n_samples, noise=0.05 )

# Incorrect number of clusters
y_pred = DBSCAN(eps=0.2).fit_predict(X)

plt.figure(2)
plt.style.use('fivethirtyeight')
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("DBSCAN Clustering Ideal Geometry")

sil = silhouette_score(X, y_pred)
print sil

""" Hierarchical Ideal Problem """
n_samples = 2500
random_state = 170
X, y = make_blobs(n_samples=n_samples, n_features=2, centers=5, cluster_std=3.0 )

# Incorrect number of clusters
y_pred = AgglomerativeClustering(n_clusters=5).fit_predict(X)

plt.figure(3)
plt.style.use('fivethirtyeight')
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Hierarchical Clustering Ideal Geometry")

from sklearn.metrics import silhouette_score
sil = silhouette_score(X, y_pred)
print sil