from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score, StratifiedKFold

    
def CrossValidateSMOTE(data, labels, clf, folds=10, runSMOTE=True):
    from unbalanced_dataset import SMOTE
    from sklearn.metrics import confusion_matrix as confmat
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    columns = []
    
    if type(data) is not np.ndarray:
        data = data.as_matrix()
        
    if type(labels) is not np.ndarray:
        labels = labels.as_matrix().ravel()
    
    skf = StratifiedKFold(labels,n_folds=folds, shuffle=False)
    sets = [{'train':train, 'test':test} for train, test in skf]
    acc = []
    fmeasure = []
    recall = []
    precision = []
    cm = np.array([0, 0, 0, 0]).reshape(2,2)
    
    for fold in sets:
        data_train = data[fold['train']]
        labels_train = labels[fold['train']]
        
    
        bugs = sum(labels_train)
        ratio = float(len(labels_train)-bugs)/bugs
        
        data_test = data[fold['test']]
        labels_test = labels[fold['test']]
        if runSMOTE:
            smote = SMOTE(ratio=ratio, verbose=False, kind='borderline1')
            data_train, labels_train = smote.fit_transform(data_train,labels_train)
        
        clf.fit(data_train, labels_train)
        hypot = clf.predict(data_test)
        
        acc.append(accuracy_score(hypot, labels_test))
        fmeasure.append(f1_score(hypot, labels_test))
        recall.append(recall_score(hypot, labels_test))
        precision.append(precision_score(hypot, labels_test))
        
        cm += confmat(labels_test, hypot)
        
    return acc, fmeasure, recall, precision, cm

  
    
        
def CrossValidateModel(model_pipe, X, y, score, cv = None, folds=10):
    """Run crossvalidation on a model (pipe) and return the resulting score
    Arguments:
        model_pipe - a pipeline including some model, as well as any preprocessing
        X, y - the data to run the model on. np.arrays, Nxd and Nx1 respectively
        score - string specifying the score to be returned from cross-validation
        cv - the crossvalidation method (default: None, StratifiedKFold will be used)
        folds - number of folds fro stratified k fold (default: 10)
    """
    #attempt to work with pandas DataFrame as well
    if type(X) is not np.ndarray: 
        try:
            X = X.as_matrix()
        except AttributeError:
            print "Attempting to transform X from", type(X), "to numpy.ndarray failed"
            raise
            
    if type(y) is not np.ndarray:
        try:
            y = y.as_matrix().ravel()
        except AttributeError:
            print "Attempting to transform y from", type(X), "to numpy.ndarray failed"
            raise
    
    if cv is None:
        skf = StratifiedKFold(y,n_folds=folds)
    
    return cross_val_score(model_pipe, X, y, cv = skf, scoring=score)