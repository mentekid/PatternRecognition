import pandas as pd
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

plt.close('all')

""" Load Data """
loadall=True #set to true to load from csv

#load from pre-saved sparse matrix (fast and sexier)
if not loadall:
    import sparse_manip
    sparsefile = 'data/data.npz' #default location
    try:
        data = sparse_manip.load(sparsefile)
    except IOError:
        loadall = True

#load from csv (slow)
if loadall:
    #load data
    filename='data/dataset.txt'
    X = pd.read_csv(filename, ';')

    #'real world' data (no indication regarding origin)
    labels = X['category'] #let's be fair, but not too fair :P
    X = X.drop('category', axis=1)
    X = X.drop('project', axis=1)


    #pandas to sparse numpy
    data = sparse.csr_matrix(X.as_matrix())
    import sparse_manip
    sparse_manip.save('data', data)

N, d = data.shape

""" Visualisation """
binary = np.zeros(data.shape)
N, d = data.shape
for i in xrange(N):
    for j in xrange(d):
        binary[i][j] = (data[i][j] > 0) #presence of term j in library i
    
    
presence = sum(binary)

#boxplot of absolute frequencies
plt.boxplot(presence)
plt.show()

#plot of absolute frequencies
plt.plot(xrange(len(presence)), sorted(presence))
plt.show()


from sklearn.metrics.pairwise import pairwise_distances
distances = pairwise_distances(data, metric='euclidean')
plt.imshow(distances); plt.title('Euclidean Similarity of initial data')
plt.colorbar()


distances = pairwise_distances(data, metric='cosine')
plt.figure(4)
plt.imshow(distances); plt.title('Cosine Similarity of initial data')
plt.colorbar()
plt.show()

""" Processing data """
from sklearn.feature_extraction.text import TfidfTransformer as tfidf
transformer = tfidf()
data = transformer.fit_transform(data)

distances = pairwise_distances(data, metric='cosine')
plt.figure(6)
plt.imshow(distances); plt.title('Cosine Similarity of tfidf data')
plt.colorbar()
plt.show()