import matplotlib.pyplot as plt
import numpy as np

""" Load data """
import data_read as dtrd
data_const, features, projects_true = dtrd.All(sparse=False)

import mypreprocessing as prp

plots = [None]*len(range(1,11))

from sklearn.cluster import KMeans
from sklearn import metrics
""" Preprocessing """

data = prp.TrimmingPresence(data_const, low_thresh=1, hig_thresh=70)  

#data = prp.RowWiseNorm(data)
#data = prp.tfidf(data)
#data = np.array(data.todense())    

data = prp.LDAProjection(data, LDA_topics=12)

silh = []
comp = []
homo = []
vmea = []
for k in range(2, 11):
    print k
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100)
    km.fit(data)
    silh.append(metrics.silhouette_score(data, km.labels_))
    comp.append(metrics.completeness_score(projects_true, km.labels_))    
    vmea.append(metrics.v_measure_score(projects_true, km.labels_))
    homo.append(metrics.homogeneity_score(projects_true, km.labels_))
    
plot = plt.figure()  
plt.plot(range(2,11), silh)
plt.plot(range(2,11), comp)
plt.plot(range(2,11), homo)
plt.plot(range(2,11), vmea)
plt.title('k-Means clustering, LDA Projection')
plt.xlabel('k clusters')
plt.ylabel('Score')
plt.legend(['silhouette', 'completeness', 'homogeneity', 'v-measure'])
plot.show()
