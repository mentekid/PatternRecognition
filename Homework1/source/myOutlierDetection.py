import numpy as np
import pandas as pd

def column_OR(alist):
    return reduce(lambda x,y: x or y, alist)

def isoutlier(x, q1, q3, OF, EVF):
    """ Returns true if x is outlier, else false"""
    IQR = q3-q1 #interquantile range
    return int((x > q3 + OF*IQR and x <= q3 + EVF*IQR) or (x >= q1 - EVF*IQR and x < q1 - OF*IQR))

def isextreme(x, q1, q3, OF, EVF):
    """ Returns true if x is extreme value, else false"""
    IQR = q3-q1 #interquantile range
    return int((x > q3+EVF*IQR) or (x < q1 - EVF*IQR))
     

def outlier_value(feature, q1, q3, OF, EVF):
    """ Returns a list with 0 in pos i if feature element i is not an outlier,
    otherwise 1"""
    feature = list(feature)
    return [isoutlier(x, q1, q3, OF, EVF) for x in feature]


def extreme_value(feature, q1, q3, OF, EVF):
    """ Returns a list with 0 in pos i if feature element i is not an extreme value,
    otherwise 1"""
    feature = list(feature)
    return [isextreme(x, q1, q3, OF, EVF) for x in feature]
    
def interquantileRange(X, OF=3, EVF=6, perFeature = False):
    """
        Creates two new features, Outlier and Extreme Value, for the data.
        Assumes a class label named bugs and ignores it
        
        Parameters
        ----------
        X : a pandas dataframe containing the data
        OF : Defines interquantile range (Outlier Factor)
        EVF : Defines extreme value range (Extreme Value Factor)
        perFeature : If set to False, only 2 new features will be added. If set
            to True, 2 features per current feature will be added
        """
    
    data = X.copy(deep=True)
    
    N, d = data.shape
    realfeatures = filter(lambda x: x != 'bugs', list(data))
    #create 2 new features for every feature in the dataset
    for feature in realfeatures:
        q1, q3 = np.percentile(data[feature].as_matrix(), [25, 75])
        data[feature+'_outlier'] = pd.Series(\
            outlier_value(data[feature], q1, q3, OF, EVF)\
            , index=data.index)
        data[feature+'_extreme'] = pd.Series(\
            extreme_value(data[feature], q1, q3, OF, EVF),\
            index=data.index)
    
    if not perFeature:
        
        """
        create a new feature as the or of all outlier features and drop
        the individual outlier features
        """
        #features to be eliminated
        outlierfeatures = [feature+'_outlier' for feature in realfeatures]
        #row-wise or on features
        outliers = data[outlierfeatures].as_matrix()
        outliers = list(np.apply_along_axis(column_OR, 1, outliers))
        #append and drop
        data['outlier'] = pd.Series(outliers, index=data.index)
        data = data.drop(outlierfeatures, 1)
        
        """
        create a new feature as the or of all extreme value features and drop
        the individual extreme value features
        """
        #features to be eliminated
        extremefeatures = [feature+'_extreme' for feature in realfeatures]
        #row-wise or on features
        extremes = data[extremefeatures].as_matrix()
        extremes = list(np.apply_along_axis(column_OR, 1, extremes))
        #append and drop
        data['extreme'] = pd.Series(extremes, index=data.index)
        data = data.drop(extremefeatures, 1)
        
        
    return data
        