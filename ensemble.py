# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:09:01 2017

@author: Gareth
"""

#%% Imports
import pandas as pd
import numpy as np
import pickle as pic
import xgboost as xgb
import matplotlib.pyplot as plt
from XGBMod import XGBHelpers

#%% Class: Ens

class ens(XGBHelpers):
    
    def __init__(self, testPaths={}, trainPaths={}, 
                 trainLabels = [], 
                 subFile = '', name='ensemble'):
        
        
        testPreds = pd.DataFrame()                  
        # Load test and train predictions 
        for k,v in testPaths.items():
            m = pd.read_csv(v)
            m.columns = ['id', k]
            
            testPreds = pd.concat([testPreds, m], axis=1)
        
                
        trainPreds = pd.DataFrame() 
        if len(trainPaths)>0:
            
                for k,v in trainPaths.items():
                    m = pd.read_csv(v)
                    m.columns = ['id', k]
                
                    trainPreds = pd.concat([trainPreds, m], axis=1)
                
                # Also load trainTable to get trainLabels
                # f = open('trainTable.p', "rb")    
                # trainTable = pic.load(f)    
                # f.close() 
                                
                
                # self.trainTable = trainTable
                self.trainPreds = trainPreds.iloc[:, trainPreds.columns != 'id']
                self.train = trainPreds
                self.trainIDs = trainLabels['name']
                self.trainNModels = len(trainPreds)
                self.trainLabels = trainLabels['cancer']
                
                # Do an ID check here - order should be the same    
                
                
        else:
            # self.trainTable = []
            self.trainPreds = []
            self.train = []
            self.trainIDs = []
            self.trainNModels = []
            self.trainLabels = []
        
        
        
        
        self.name = name

        self.test = testPreds
        self.testIDs = testPreds['id']
        self.testPreds = testPreds.iloc[:, testPreds.columns != 'id']
        self.subFile = pd.read_csv(subFile)
        
            
    
    def reducePreds(self, method = 'mean', testWeights={}):
        
        if method=='mean':
            if len(testWeights)==0:
                self.redPreds = self.testPreds.mean(axis=1)
            else:
                tmp = self.testPreds
                tmp2 = pd.DataFrame()
                for k,v in testWeights.items():
                    print(k,v)
                    # Multiply columm by weight
                    tmp2[k] = tmp[k]*v

                self.redPreds = tmp2.sum(axis=1)
                    
        elif method=='weighted':
            pass
            
        elif method=='xgbtree':
            
            # Train new model
            nTotal = self.trainPreds.shape[0]
            validProp = 0.2
            
            rowIdx = np.random.choice(range(0, nTotal), nTotal, replace=False)
            splitLoad = round(nTotal*(0-validProp))
            
            trainIdx = rowIdx[0:splitLoad]
            validIdx = rowIdx[splitLoad:nTotal]

            trainData = np.array(self.trainPreds.iloc[trainIdx,:])
            trainLabels = np.array(self.trainLabels.iloc[trainIdx])
            validData = np.array(self.trainPreds.iloc[validIdx,:])
            validLabels = np.array(self.trainLabels.iloc[validIdx])

            dTrain = xgb.DMatrix(trainData, label=trainLabels, 
                                 feature_names=self.trainPreds.columns)
            dValid = xgb.DMatrix(validData, label=validLabels, 
                                 feature_names=self.trainPreds.columns)
            
            param = {'booster':'gbtree',
             'bst:max_depth': 2,
             'subsample': 1,
             'reg_lamdba': 25,
             'reg_alpha': 25,
             'bst:eta': 0.0005, 
             # 'scale_pos_weight': np.sum(trainLabels==1)/np.sum(trainLabels==0),
             'silent': 0, 
             'objective' : 'binary:logistic', 
             }
            param['nthread'] = 3
            param['eval_metric'] = ['auc', 'logloss']
            
            evallist  = [(dValid,'eval'), (dTrain,'train')]
            
            num_round = 1000
            bst = xgb.train(param, dTrain, num_round, evallist)
            
            xgb.plot_importance(bst)
            plt.show()
            self.traiModelImp = bst.get_fscore()
            
            # Predict from new model
            # From train for comparison
            trainData = np.array(self.trainPreds)
            trainLabels = np.array(self.trainLabels)

            dTrain = xgb.DMatrix(trainData, label=trainLabels, feature_names=self.trainPreds.columns)
            self.trainRedPreds = bst.predict(dTrain)            
            
            # From test for output
            dTest = xgb.DMatrix(self.testPreds, feature_names=self.trainPreds.columns)
            self.redPreds = bst.predict(dTest)
            
            # plt.plot(self.redPreds)
            # plt.show()
            
    def plot(self, oldPreds, newPreds, labels=[]):
        plt.figure(figsize =(16,4))
        
        if len(labels)>0:
            plt.plot(range(0, len(labels)), labels, color='k')
        
       
        plt.plot(oldPreds, color='b')        
        plt.plot(newPreds, color='r')
        
       
        plt.show()
        
    
    def writeSub(self):
        
        out = self.subFile
        out['cancer'] = self.redPreds
        out.to_csv('Predictions\\'+self.name+'.csv', index=False)
        print('Submission saved')
     
