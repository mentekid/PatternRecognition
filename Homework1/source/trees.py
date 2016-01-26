from __future__ import division
from sklearn import tree #decision tree
import pandas as pd #dataframes, csvread, etc
import numpy as np #matlab-like arrays


def shuffle_in_unison_inplace(a, b):
    shuf = np.random.permutation(a.index)
    a=a.reindex(shuf)
    b=b.reindex(shuf)
    return (a,b)
    
def trainandtest(data, labels):

    (N, d) = data.shape
    """
    Section: Separate training/testing data
    """
    thr = int(round(0.70*N))
    train_data = data[0:thr]

    train_labels = labels[0:thr]

    test_data = data[thr:]
    test_labels = labels[thr:]
    
    """
    Section: decision tree - gini and entropy-based splitting
    """

    #gini splitting criterion
    giniTree = tree.DecisionTreeClassifier(min_samples_split=10, criterion='gini')
    giniTree.fit(train_data, train_labels)

    hypot = giniTree.predict(test_data)
    acc_gini = float(sum([h==true for (h, true) in zip(hypot, test_labels.as_matrix().ravel())])*100)/float(len(hypot))

    entropyTree = tree.DecisionTreeClassifier(min_samples_split=20, criterion='entropy')
    entropyTree.fit(train_data, train_labels)

    hypot = entropyTree.predict(test_data)
    acc_entropy = float(sum([h==true for (h, true) in zip(hypot, test_labels.as_matrix().ravel())])*100)/float(len(hypot))
    
    return (acc_gini,acc_entropy)

def crossvalidate(data, labels):
    from GeneralCrossValidation import CrossValidateModel
    giniTree = tree.DecisionTreeClassifier(min_samples_split=40, criterion='gini')
    entropyTree = tree.DecisionTreeClassifier(min_samples_split=40, criterion='entropy')
    
    gini = CrossValidateModel(giniTree, data, labels, 'accuracy')
    entropy = CrossValidateModel(entropyTree, data, labels, 'accuracy')
    
    return gini, entropy

def averageTrainTest():
    datasetFile = 'data/source-code-metrics_train.csv'
    labelsFile = 'data/bugs_train.csv'
    data = pd.read_csv(datasetFile, ';') #separate at semicolon instead of comma
    labels = pd.read_csv(labelsFile, ';')
    data.set_index('classid',inplace=True)
    labels.set_index('classid',inplace=True)
    
    """
    Section: SMOTE for class balance
    """ 
    from unbalanced_dataset import SMOTE #, TomekLinks

    columns = list(data)
    smote = SMOTE(ratio=3, verbose=False, kind='regular')
    smox,smoy = smote.fit_transform(data.as_matrix(),labels.as_matrix().ravel())
    data = pd.DataFrame(smox, columns=columns)
    labels = pd.DataFrame(smoy, columns=['bugs'])

    
    """
    Section: outlier detection
    """
    from myOutlierDetection import interquantileRange
    interquantileRange(data, perFeature = False)
    
    data = [trainandtest(data, labels) for _ in range(500)]
    return (sum([data[i][0] for i in range(len(data))])/len(data),sum([data[i][1] for i in range(len(data))])/len(data))
    
def runCrossValidation(runSMOTE = True, runIQR = True):
    datasetFile = 'data/source-code-metrics_train.csv'
    labelsFile = 'data/bugs_train.csv'
    data = pd.read_csv(datasetFile, ';') #separate at semicolon instead of comma
    labels = pd.read_csv(labelsFile, ';')
    data.set_index('classid',inplace=True)
    labels.set_index('classid',inplace=True)
    
    if runSMOTE:
        """
        Section: SMOTE for class balance
        """ 
        from unbalanced_dataset import SMOTE #, TomekLinks
    
        columns = list(data)
        smote = SMOTE(ratio=3, verbose=False, kind='regular')
        smox,smoy = smote.fit_transform(data.as_matrix(),labels.as_matrix().ravel())
        data = pd.DataFrame(smox, columns=columns)
        labels = pd.DataFrame(smoy, columns=['bugs'])

    if runIQR:
        """
        Section: outlier detection
        """
        from myOutlierDetection import interquantileRange
        interquantileRange(data, perFeature = False)
    
    return crossvalidate(data.as_matrix(), labels.as_matrix().ravel())