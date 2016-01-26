# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:53:53 2016

@author: et3rn1ty
"""

def SparseData(filename = 'data/drop_presence.npz'):
    import sparse_manip
    data = sparse_manip.load(filename)
    return data

def DenseData(filename = 'data/dataset.txt'):
    import pandas as pd    
    data = pd.read_csv(filename, ';')
    
    data = data.drop('category', axis=1)
    data = data.drop('project', axis=1)
    
    data =data.as_matrix()
    return data
   
def Features(filename = 'data/terms.csv'):
    features = open(filename).read().splitlines()
    return features
    
def Projects(filename = 'data/project_names.csv'):
    projects = open(filename).read().splitlines()
    return projects
    
def All(data = None, sparse=True):    
    if sparse:
        if data==None: 
            data = 'data/drop_presence.npz'
        return (SparseData(data), Features(), Projects())
    else:
        if data==None:
            data = 'data/dataset.txt'
        return (DenseData(data), Features(), Projects())
        
