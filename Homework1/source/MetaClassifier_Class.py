import numpy as np
import pandas as pd

class MetaClassifier:
    """
    Implements a meta-classifier using an ensemble of classifiers to create a
    meta-model. An aggregating classifier is trained on the meta-model and
    makes predictions on data by using the classification result of each
    classifier in the ensemble.
    """
    
    def __init__(self, ensemble, aggregator, useSMOTE=False):
        """
        Creates and returns the metaclassifier's ensemble and aggregator
        """
        self.required_methods = ['fit', 'predict']
        for method in self.required_methods:
            if method not in dir(aggregator):
                print "Provided aggregator doesn't have one or more desired methods"
                raise AttributeError
            for clf in ensemble:
                if method not in dir(clf):
                    print "A provided classifier doesn't have one or more desired methods"
                    raise AttributeError
                
        self.aggregator = aggregator
        self.ensemble = ensemble
        self.useSMOTE = useSMOTE
        
    def fit(self, data, labels):
        """
        Training (fitting) the meta-classifier requires training each individual
        classifier in the ensemble and using testing data that has not been used
        in the classifiers' training set to train the meta-classifier.
        To do this, we use 10-fold Stratified Cross-Validation to produce a
        training set for the meta-classifier equal to the one provided.
        
        Arguments
        ---------
        data: pandas (N,d) dataframe with data to be trained on
        labels: pandas (N,1) dataframe with labels for training data
        SMOTE: perform SMOTE as part of cross-validation to balance classes
        """
        from unbalanced_dataset import SMOTE
        from sklearn.cross_validation import StratifiedKFold
        
        #if not isinstance(data, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        #    print "data and labels must be in pandas DataFrame form"
        #    raise TypeError
        N, d = data.shape
        
        self.data_train = np.copy(data)#.copy(deep=True)
        self.labels_train = np.copy(labels)#.copy(deep=True)

        #training data for metaclassifier (results of each classifier in ensemble)
        self.fusion_data = pd.DataFrame() #(fusion_labels = labels_train!)
        
        skf = StratifiedKFold(self.labels_train, n_folds=10)
        sets = [{'train':train, 'test':test} for train, test in skf]
        
        count = 0
        for clf in self.ensemble:
            hypothesis = list()
            for fold in sets:
                
                #separate training/testing set for fold, use SMOTE if asked to
                data_train_fold = self.data_train[fold['train']]
                labels_train_fold = self.labels_train[fold['train']]
            
                if self.useSMOTE:
                    bugs = sum(labels_train_fold)
                    ratio = float(len(labels_train_fold)-bugs)/bugs
                    smote = SMOTE(ratio=ratio, verbose=False, kind='borderline1')
                    data_train_fold, labels_train_fold = smote.fit_transform(data_train_fold,labels_train_fold)
                    #data_train_fold = pd.DataFrame(data_train_fold, columns=categories)
                    #labels_train_fold = pd.DataFrame(labels_train_fold, columns=['bugs'])
                
                #fit the classifier with the training data of current fold
                clf.fit(data_train_fold, labels_train_fold)
                
                #make a prediction with the testing data of current fold
                data_test_fold = self.data_train[fold['test']]
                y = clf.predict(data_test_fold)
                
                #store data for the meta-classifier
                hypothesis.extend(list(y))
                
            #re-train the model using the entire available data (better performance)
            if self.useSMOTE:
                    bugs = sum(self.labels_train)
                    ratio = float(len(self.labels_train)-bugs)/bugs
                    smote = SMOTE(ratio=ratio, verbose=False, kind='borderline1')
                    data_train_clf, labels_train_clf = smote.fit_transform(self.data_train, self.labels_train)
                    #data_train_fold = pd.DataFrame(data_train_fold, columns=categories)
                    #labels_train_fold = pd.DataFrame(labels_train_fold, columns=['bugs'])
                
            
            clf.fit(data_train_clf, labels_train_clf)
            
            #new column of metaclassifier training data (this classifier's hypothesis)
            self.fusion_data['classifier_'+str(count)] = np.array(hypothesis)
            count+=1
        
        #perform smote on the fusion data to even out the classes
        if self.useSMOTE:
            columns = list(self.fusion_data)
            bugs = sum(self.labels_train)
            ratio = float(len(self.labels_train)-bugs)/bugs
            smote = SMOTE(ratio=ratio, verbose=False, kind='borderline1')
            self.fusion_data, self.labels_train = smote.fit_transform(self.fusion_data.as_matrix(),self.labels_train)
            self.fusion_data = pd.DataFrame(self.fusion_data, columns=columns)
        
        #train the aggregator using the fusion set created earlier
        self.aggregator.fit(self.fusion_data.as_matrix(), self.labels_train)
        return    
        
    
    def predict(self, data):
        """
        Uses the data and the classifier ensemble to produce input to the
        trained meta-classifier, then uses that input to produce predictions
        for the data.
        """
        input_data = pd.DataFrame()
        for count, clf in enumerate(self.ensemble):
            y = clf.predict(data)
            input_data['classifier_'+str(count)] = y
        
        return self.aggregator.predict(input_data.as_matrix())