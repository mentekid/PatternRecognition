# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5

@authors: Yannis Mentekidis, Themis Papavasileiou, Panos Siatos
Includes all preprocessing routines found useful for the clustering task of
Homework 2
"""

def LDAProjection(data, features=None, LDA_topics=20, verbose=False):
    """
    Creates a projection of the dataset on a new feature space discovered by
    Latent Dirichlet Allocation.
    Requires a dense matrix for data.
    """
    import lda
    import numpy as np
    
    model = lda.LDA(n_topics=LDA_topics,n_iter=1000,random_state=1)
    model.fit_transform(data)
    
    if verbose and features != None:
        #print topic words
        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(features)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
    
    
    """ Transform data into 80 libraries and their projections on the LDA topics"""
    data = np.dot(data, np.transpose(model.topic_word_))
    return data
    
def TrimmingPresence(data, low_thresh=2, hig_thresh=70):
    """
    Removes features present in too many or too few libraries.
    Requires a dense data matrix
    """
    import numpy as np
    binary = np.zeros(data.shape)
    N, d = data.shape
    for i in xrange(N):
        for j in xrange(d):
            binary[i][j] = (data[i][j] > 0) #presence of term j in library i
    
    
    presence = sum(binary)
    
    drop_hig = [x for (x, y) in zip(range(109706), presence)  if y > hig_thresh]
    drop_low = [x for (x, y) in zip(range(109706), presence)  if y < low_thresh]
    
    data = np.delete(data, drop_hig, axis=1)
    data = np.delete(data, drop_low, axis=1)
    return data
    
def RowWiseNorm(data):
    """
    Normalizes features row-wise, so that large libraries don't give too much
    weight to their features so that smaller libraries can participate equally
    """
    from sklearn.preprocessing import normalize
    #axis = 1 for row-wise, norm = l1 for frequency per library
    data = normalize(data.astype(float), axis=1, norm='l1')
    return (data)
    

def tfidf(data):
    """
    one-click tfidf feature creation
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    data = transformer.fit_transform(data)
    return data
    
def PCA(data, n_components):
    """
    one-click Principal Components Analysis
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components = n_components)
    data = pca.fit_transform(data)
    return data