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
                
                # TODO:
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
        
        # Update to allow for no train preds
        if method=='mean':
            if len(testWeights)==0:
                self.redPreds = self.testPreds.mean(axis=1)
            else:
                tmp = self.testPreds
                tmp2 = pd.DataFrame()
                
                trainTmp = self.trainPreds
                trainTmp2 = pd.DataFrame()
                for k,v in testWeights.items():
                    print(k,v)
                    # Multiply columm by weight
                    tmp2[k] = tmp[k]*v
                    
                    trainTmp2[k] =trainTmp[k]*v

                self.redPreds = tmp2.sum(axis=1)
                self.trainRedPreds = trainTmp2.sum(axis=1)
                
                self.writeSub()
                self.savePreds()
                    
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
            self.writeSub()
            self.savePreds()
            
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
        out.to_csv('Predictions\\TEST_'+self.name+'.csv', index=False)
        print('Submission saved')
   
    
    def savePreds(self):
        """ 
        Reduce predictions
        Save predictions alongside ID
        """
        
        trainSub = pd.DataFrame()
        trainSub['id'] = self.train['id'].iloc[:,0]
        trainSub['cancer'] = self.trainRedPreds
        trainSub.to_csv('Predictions\\TRAIN_'+self.name+'.csv', index=False) 
   
   
#%% Templates
   
class ensTemplates():
    
    def XGBs_V1(sFile):
        """
        0.55-0.57
        """
        testPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TEST_S2V1XGBTree.csv',
                     'GenFeaS2V1_XGBDart' : 'Predictions\\TEST_S2V1XGBDart.csv',
                     }
                     
        trainPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TRAIN_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TRAIN_S2V1XGBDart.csv',
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V1.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
                     
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V1')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
      
    
    def XGBs_V2(sFile):
        """
        genFea V1 and V3, XGB tree and dart 0.51-0.55
        """
        testPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TEST_S2V1XGBTree.csv',
                     'GenFeaS2V1_XGBDart' : 'Predictions\\TEST_S2V1XGBDart.csv',
                     'GenFeaS2V3_XGBTree' : 'Predictions\\TEST_S2V3XGBTree.csv',
                     'GenFeaS2V3_XGBDart' : 'Predictions\\TEST_S2V3XGBDart.csv'
                     }
                     
        trainPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TRAIN_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TRAIN_S2V1XGBDart.csv',
                      'GenFeaS2V3_XGBTree' : 'Predictions\\TRAIN_S2V3XGBTree.csv',
                      'GenFeaS2V3_XGBDart' : 'Predictions\\TRAIN_S2V3XGBDart.csv'
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V2')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
        
        
    def XGBs_V3(sFile):
        """
        genFea V1 and V3, XGB tree and dart and TFV1, TFV1, UNET + BN. 
        """
        testPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TEST_S2V1XGBTree.csv',
                     'GenFeaS2V1_XGBDart' : 'Predictions\\TEST_S2V1XGBDart.csv',
                     'GenFeaS2V3_XGBTree' : 'Predictions\\TEST_S2V3XGBTree.csv',
                     'GenFeaS2V3_XGBDart' : 'Predictions\\TEST_S2V3XGBDart.csv',
                     'TF3D_1_BN_P1' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P1.csv',
                     'TF3D_1_BN_P2' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P2.csv',
                     'TF3D_1_BN_P3' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P3.csv',
                     'TF3D_1_BN_P4' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P4.csv',
                     'TF3D_1_BN_P5' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P5.csv',
                     'TF3D_2_BN_P1' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P1.csv',
                     'TF3D_2_BN_P2' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P2.csv',
                     'TF3D_2_BN_P3' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P3.csv',
                     'TF3D_2_BN_P4' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P4.csv',
                     'TF3D_2_BN_P5' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P5.csv',
                     'TF3D_1_UN_P1' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P1.csv',
                     'TF3D_1_UN_P2' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P2.csv',
                     'TF3D_1_UN_P3' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P3.csv',
                     'TF3D_1_UN_P4' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P4.csv',
                     'TF3D_1_UN_P5' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P5.csv',
                     'TF3D_2_UN_P1' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P1.csv',
                     'TF3D_2_UN_P2' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P2.csv',
                     'TF3D_2_UN_P3' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P3.csv',
                     'TF3D_2_UN_P4' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P4.csv',
                     'TF3D_2_UN_P5' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P5.csv'
                     }
                     
        trainPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TRAIN_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TRAIN_S2V1XGBDart.csv',
                      'GenFeaS2V3_XGBTree' : 'Predictions\\TRAIN_S2V3XGBTree.csv',
                      'GenFeaS2V3_XGBDart' : 'Predictions\\TRAIN_S2V3XGBDart.csv',
                      'TF3D_1_BN_P1' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P1.csv',
                      'TF3D_1_BN_P2' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P2.csv',
                      'TF3D_1_BN_P3' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P3.csv',
                      'TF3D_1_BN_P4' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P4.csv',
                      'TF3D_1_BN_P5' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P5.csv',
                      'TF3D_2_BN_P1' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P1.csv',
                      'TF3D_2_BN_P2' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P2.csv',
                      'TF3D_2_BN_P3' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P3.csv',
                      'TF3D_2_BN_P4' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P4.csv',
                      'TF3D_2_BN_P5' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P5.csv',
                      'TF3D_1_UN_P1' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P1.csv',
                      'TF3D_1_UN_P2' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P2.csv',
                      'TF3D_1_UN_P3' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P3.csv',
                      'TF3D_1_UN_P4' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P4.csv',
                      'TF3D_1_UN_P5' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P5.csv',
                      'TF3D_2_UN_P1' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P1.csv',
                      'TF3D_2_UN_P2' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P2.csv',
                      'TF3D_2_UN_P3' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P3.csv',
                      'TF3D_2_UN_P4' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P4.csv',
                      'TF3D_2_UN_P5' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P5.csv'
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V3')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
        
    def XGBs_V4(sFile):
        """
        Mean of basic node models/predictions
        """
        testPaths = {
                     'TF3D_1_BN_P1' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P1.csv',
                     'TF3D_1_BN_P2' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P2.csv',
                     'TF3D_1_BN_P3' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P3.csv',
                     'TF3D_1_BN_P4' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P4.csv',
                     'TF3D_1_BN_P5' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P5.csv',

                     }
                     
        trainPaths = {
                      'TF3D_1_BN_P1' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P1.csv',
                      'TF3D_1_BN_P2' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P2.csv',
                      'TF3D_1_BN_P3' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P3.csv',
                      'TF3D_1_BN_P4' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P4.csv',
                      'TF3D_1_BN_P5' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P5.csv',

                     }
        
        testWeights = { 'TF3D_1_BN_P1' : 0.2, 
                        'TF3D_1_BN_P2' : 0.2,
                        'TF3D_1_BN_P3' : 0.2,
                        'TF3D_1_BN_P4' : 0.2, 
                        'TF3D_1_BN_P5' : 0.2, }         
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V4_mean')
        ensemble.reducePreds('mean',testWeights=testWeights)
        ensemble.writeSub()
        
        return ensemble
        
        
    def XGBs_V5(sFile):
        testPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TEST_S2V1XGBTree.csv',
                     'GenFeaS2V1_XGBDart' : 'Predictions\\TEST_S2V1XGBDart.csv',
                     'GenFeaS2V3_XGBTree' : 'Predictions\\TEST_S2V3XGBTree.csv',
                     'GenFeaS2V3_XGBDart' : 'Predictions\\TEST_S2V3XGBDart.csv',
                     'XGBs_V4_mean' : 'Predictions\\TEST_XGBs_V4_mean.csv'
                     }
                     
        trainPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TRAIN_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TRAIN_S2V1XGBDart.csv',
                      'GenFeaS2V3_XGBTree' : 'Predictions\\TRAIN_S2V3XGBTree.csv',
                      'GenFeaS2V3_XGBDart' : 'Predictions\\TRAIN_S2V3XGBDart.csv',
                      'XGBs_V4_mean' : 'Predictions\\TRAIN_XGBs_V4_mean.csv'
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V5')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
    

      
    def XGBs_V7(sFile):
        """
        Combined BN - eval LL ~ 0.58-0.60
        """
        testPaths = {'TF3D_1_BN_P1' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P1.csv',
                     'TF3D_1_BN_P2' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P2.csv',
                     'TF3D_1_BN_P3' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P3.csv',
                     'TF3D_1_BN_P4' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P4.csv',
                     'TF3D_1_BN_P5' : 'Predictions\\TEST_TF3D_40_BasicNodes_V1_P5.csv',
                     'TF3D_2_BN_P1' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P1.csv',
                     'TF3D_2_BN_P2' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P2.csv',
                     'TF3D_2_BN_P3' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P3.csv',
                     'TF3D_2_BN_P4' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P4.csv',
                     'TF3D_2_BN_P5' : 'Predictions\\TEST_TF3D_40_BasicNodes_V2_P5.csv',
                     }
                     
        trainPaths = {'TF3D_1_BN_P1' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P1.csv',
                      'TF3D_1_BN_P2' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P2.csv',
                      'TF3D_1_BN_P3' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P3.csv',
                      'TF3D_1_BN_P4' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P4.csv',
                      'TF3D_1_BN_P5' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V1_P5.csv',
                      'TF3D_2_BN_P1' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P1.csv',
                      'TF3D_2_BN_P2' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P2.csv',
                      'TF3D_2_BN_P3' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P3.csv',
                      'TF3D_2_BN_P4' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P4.csv',
                      'TF3D_2_BN_P5' : 'Predictions\\TRAIN_TF3D_40_BasicNodes_V2_P5.csv',
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V7')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
        
    def XGBs_V8(sFile):
        """
        UN - 0.57-0.59
        """
        testPaths = {'TF3D_1_UN_P1' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P1.csv',
                     'TF3D_1_UN_P2' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P2.csv',
                     'TF3D_1_UN_P3' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P3.csv',
                     'TF3D_1_UN_P4' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P4.csv',
                     'TF3D_1_UN_P5' : 'Predictions\\TEST_TF3D_40_UNETNodes_V1_P5.csv',
                     'TF3D_2_UN_P1' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P1.csv',
                     'TF3D_2_UN_P2' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P2.csv',
                     'TF3D_2_UN_P3' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P3.csv',
                     'TF3D_2_UN_P4' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P4.csv',
                     'TF3D_2_UN_P5' : 'Predictions\\TEST_TF3D_40_UNETNodes_V2_P5.csv'
                     }
                     
        trainPaths = {'TF3D_1_UN_P1' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P1.csv',
                      'TF3D_1_UN_P2' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P2.csv',
                      'TF3D_1_UN_P3' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P3.csv',
                      'TF3D_1_UN_P4' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P4.csv',
                      'TF3D_1_UN_P5' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V1_P5.csv',
                      'TF3D_2_UN_P1' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P1.csv',
                      'TF3D_2_UN_P2' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P2.csv',
                      'TF3D_2_UN_P3' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P3.csv',
                      'TF3D_2_UN_P4' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P4.csv',
                      'TF3D_2_UN_P5' : 'Predictions\\TRAIN_TF3D_40_UNETNodes_V2_P5.csv'
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V8')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble
        
        
    def XGBs_V9(sFile):
        """
  
        """
        testPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TEST_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TEST_S2V1XGBDart.csv',
                      'GenFeaS2V3_XGBTree' : 'Predictions\\TEST_S2V3XGBTree.csv',
                      'GenFeaS2V3_XGBDart' : 'Predictions\\TEST_S2V3XGBDart.csv',
                      'XGBs_V7' : "Predictions\\TEST_XGBs_V7.csv"
                     }
                     
        trainPaths = {'GenFeaS2V1_XGBTree' : 'Predictions\\TRAIN_S2V1XGBTree.csv',
                      'GenFeaS2V1_XGBDart' : 'Predictions\\TRAIN_S2V1XGBDart.csv',
                      'GenFeaS2V3_XGBTree' : 'Predictions\\TRAIN_S2V3XGBTree.csv',
                      'GenFeaS2V3_XGBDart' : 'Predictions\\TRAIN_S2V3XGBDart.csv',
                      'XGBs_V7' : "Predictions\\TRAIN_XGBs_V7.csv"
                     }
        
        # Get the train labels from one of the feature tables
        trainTable = ens.load('trainTableS2V3.p')   
        trainLabels = trainTable.iloc[:,0:2]
        
        # Create ensemble
        ensemble = ens(testPaths=testPaths, trainPaths=trainPaths, 
                       trainLabels=trainLabels,
                       subFile = sFile, name='XGBs_V9')
        ensemble.reducePreds('xgbtree')
        ensemble.writeSub()
        
        return ensemble

