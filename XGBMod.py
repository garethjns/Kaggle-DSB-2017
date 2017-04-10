# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:37:37 2017

@author: Gareth
"""

#%% Imports

import xgboost as xgb
import pickle as pic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


#%% Class: XGBHelpers

class XGBHelpers():
    """
    Basic load methods, etc. Used by XGBMod here and ensemble later.
    """
    @staticmethod
    def load(fn):
        """
        Load pickeled model if it exists.
        """
        if os.path.isfile(fn):
            f = open(fn, "rb")    
            data = pic.load(f)    
            f.close()    
        else:
            data = None
            
        return data
        
        
    @staticmethod
    def splitValid(data, vProp=0.2):
        """
        Randomly split validation set off of training set.
        """
        nTotal = data.shape[0]
        rowIdx = np.random.choice(range(0, nTotal), nTotal, replace=False)
        splitLoad = round(nTotal*(0-vProp))
        trainIdx = rowIdx[0:splitLoad]
        validIdx = rowIdx[splitLoad:nTotal]

        return trainIdx, validIdx 
    
    
    @staticmethod
    def checkFeatureNames(cols):
        """ Set feature names:
        Check for duplicates - likely to be a bug in column naming rather
        than a genuine duplicate
        """
        a = pd.DataFrame({'colNames' : cols})
        dupes = a.set_index('colNames').index.get_duplicates()
        if len(dupes)>0:
            # There are duplicates
            print('Warning: "duplicate" features: ')
            print(dupes)
            
        # Append col number to names, even if there aren't any duplicates
        newNames = []
        for i in range(0,len(cols)):
            newNames.append(cols[i] + '_' + str(i+1))
            
        return newNames
    
    
    @staticmethod
    def prepTrain(data, trainIdx, validIdx):
        """
        Split training set in to train/valid
        Drop labels, Add to XGB mats
        Both test and train have [name, cancer] as first two columns, it's
        just a placeholder in test
        """
        
        # data, drop [name, caner]
        trainData = data.iloc[trainIdx,2:]
        trainLabels = data.iloc[trainIdx,1]
        validData = np.array(data.iloc[validIdx,2:])
        validLabels = np.array(data.iloc[validIdx,1])

        # Check and set feature names
        newNames = XGBMod.checkFeatureNames(trainData.columns)
        
        print(newNames)
        # Create XGB mats        
        dTrain = xgb.DMatrix(np.array(trainData), label=np.array(trainLabels), 
                             feature_names = newNames)
        dValid = xgb.DMatrix(np.array(validData), label=np.array(validLabels), 
                             feature_names = newNames)

        return dTrain, dValid
        

#%% Class: XGBMod

class XGBMod(XGBHelpers):
    """
    Trains two XGB models (Tree and Dart), saves in self.models
    Predicts training and test data (if available) for each model.
    Saves an individual submission for test data, and saves preds for
    training to use with ensemble.
    """
    def __init__(self, sampleSubPath, dataPath='', name=''):
        """
        TODO:
        Add model param setting - using template methods at the moment
        """
        
        trainPath = dataPath + 'trainTable' + name + '.p'
        testPath = dataPath + 'testTable' + name + '.p'
        
        trainData = XGBMod.load(trainPath)
        testData = XGBMod.load(testPath)   
        
        self.trainPath = trainPath
        self.testPath = testPath
        self.data = {'train' : trainData,
                     'test' : testData}
        
        self.models = {'XGBTree' : dict(),
                       'XGBDart' : dict()}

        self.sampleSub = sampleSubPath
        self.name = name


    def runAll(self):
        """
        Preprocess loaded data, fit XGBTree and XGBDart models, make 
        predictions on training and test sets, save to disk as .csv.
        """
        # Run PP
        self = self.PP()
        
        # For both models
        for t in ['XGBTree', 'XGBDart']:
            # Run training
            self.models[t] = self.train(self.data['train'], mod=t) 
                                                
            # Predict
            # Training data
            self.models[t]['trainPreds'] = self.predict(self.data['train'], 
                                                        mod=t)
            # Test data
            self.models[t]['testPreds'] = self.predict(self.data['test'],
                                                        mod=t)

            # Save
            # Training preds
            self.savePreds(ID = self.data['train']['name'],
                           preds = self.models[t]['trainPreds'], 
                           name = 'TRAIN_'+self.name+t)
            # Individual Submission (test)
            self.saveSub(preds = self.models[t]['testPreds'], 
                         name = 'TEST_'+self.name+t)
        

    def PP(self):
        """
        Pre process features:
        - inf, -inf -> NaN
        - NaN -> 0
        TODO:
        Add scaling?
        """
        # PP available sets
        for k,v in self.data.items():
            if v is  None:
                continue
            
            # Clear infs and NaNs
            self.data[k] = self.data[k].replace([np.inf, -np.inf], np.nan)
            self.data[k] = self.data[k].fillna(0)
        
        return self
        
       
    def train(self, data, mod='XGBTree'):
        """
        Run training.
        """
        print('Training: ' + mod)
        
        # Split training set in to tain/valid
        trainIdx, validIdx = XGBMod.splitValid(data, 
                                               vProp=0.2)
        dTrain, dValid = self.prepTrain(data,
                                  trainIdx, validIdx)
        
        # Set parameters
        if mod == 'XGBTree':
            params = XGBMod.templateXGBTree()
        elif mod == 'XGBDart':
            params = XGBMod.templateXGBDart()
        
        # Fit model
        evallist  = [(dValid,'eval'), (dTrain,'train')]
        bst = xgb.train(params, dTrain, params['nRounds'], evallist)
        
        # Plot feature importance
        xgb.plot_importance(bst)
        imp = bst.get_fscore()
        
        mod = {}
        mod = {'model': bst,
                             'fImp' : imp,
                             'params' : params,
                             'testPreds' : [],
                             'trainPreds' : []}
                             
        return mod
        
   
    def predict(self, data, mod):
        """
        Predict from model specified in type.
        """
        
        print('Predicting: ' + mod)
        
        # Check and set feature names
        newNames = XGBMod.checkFeatureNames(data.columns[2:])        
        
        # Both test and train have [name, cancer] as first two columns, it's
        # just a placeholder in test
        dTest = xgb.DMatrix(np.array(data.iloc[:,2:]), 
                                     feature_names = newNames)
        bst = self.models[mod]['model']
        
        # Predict test
        preds = bst.predict(dTest)
        
        return preds
        
        
    def saveSub(self, preds, name):
        """
        Save a submission file for stage 2
        """
        # Save stage 2 submission
        submission = pd.read_csv(self.sampleSub)
        
        submission.cancer = preds
        
        fn = 'Predictions\\'+name+'.csv'
        print('Saving: ' + fn)
        submission.to_csv(fn, index=False)     
    
        
    def savePreds(self, ID, preds, name):
        """
        Save preds that aren't for stage 2 submission.
        """
        # Save predictions wihtout using submission file
        # eg. to save training preds
        trainSub = pd.DataFrame()
        trainSub['id'] = ID
        trainSub['cancer'] = preds

        fn = 'Predictions\\'+name+'.csv'
        print('Saving: ' + fn)
        trainSub.to_csv(fn, index=False)     

        
    def templateXGBTree():
        """
        Set template params for XGBTree. All params set here for now.
        """
        params = {'booster':'gbtree',
         'bst:max_depth': 2,
         'subsample': 1,
         'reg_lamdba': 25,
         'reg_alpha': 25,
         'bst:eta': 0.0005, 
         # 'scale_pos_weight': np.sum(trainLabels==1)/np.sum(trainLabels==0),
         'silent': 0, 
         'objective' : 'binary:logistic', 
         'nThread' : 3,
         'eval_metric' : ['auc', 'logloss'],
         'nRounds' : 200}
         
        return params
         
        
    def templateXGBDart ():
        """
        Set template params for XGBDart. All params set here for now.
        """
        params = {'booster':'dart',
         'max_depth': 20,
         'subsample': 1,
         'reg_lamdba': 60,
         'reg_alpha': 60,
         'bst:eta': 0.0005, 
         'rate_drop': 0.1,
         'skip_drop': 0.5,
         # 'scale_pos_weight': np.sum(trainLabels==1)/np.sum(trainLabels==0),
         'silent': False, 
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'objective' : 'binary:logistic', 
         'nThread' : 3,
         'eval_metric' : ['auc', 'logloss'],
         'nRounds' : 300}

        return params
        