# -*- coding: utf-8 -*-
"""

# Just add evrything here for now, split later

# PPV1 running for stage 2
# PPV6 waiting
# UNET in progress
# XGB in progress
# TF not added yet
# ensemble not added yet

"""

#%% Imports

from setPaths import paths
from preprocessing import ppV6, ppV1
from UNET import UNET
from extractFeatures import features
from XGBMod import XGBMod
from ensemble import ens

import os
from glob import glob
import pandas as pd

# Auto reload functions when testing
%load_ext autoreload
%autoreload 2

processStage = 2


#%% PPV1: Set paths 

allPaths = paths()
if processStage==1:
    # Stage 1
    paRaw = allPaths.paths['Raw']
    paPP = allPaths.paths['PPed']
elif processStage==2:
    # Stage 2
    paRaw = allPaths.paths['Raw_s2']
    paPP = allPaths.paths['PPed_s2']


#%% PPV1: Load and plot one patient
      
# 55 is upside down      
p = 912 # Index  
p = 1
patients = os.listdir(paRaw)

params = {'plotHist' : True,
          'plot3D' : True,
          'plotMid2D' : True,
          'forcePP' : False}

# Create object, process one patient
PP = ppV1(paRaw, paPP, params)
PP.process(patients[p])

# Loading example - now in ppV1.verify
ppV1.verify(paPP+patients[p]+'.npz')


#%% PPV1: Process all patients
      
patients = os.listdir(paRaw)

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : True,
          'forcePP' : False}

# Create object, process all patients
PP = ppV1(paRaw, paPP, params)
PP.process(patients)


#%% PPV6: Set paths for ppV6

allPaths = paths()
if processStage==1:
    # Stage 1
    paRaw = allPaths.paths['Raw']
    paPP = allPaths.paths['PPedV6']
elif processStage==2:
    # Stage 2
    paRaw = allPaths.paths['Raw_s2']
    paPP = allPaths.paths['PPedV6_s2']


#%% PPV6: Run for one patient

params = {'plotHist' : True,
          'plot3D' : True,
          'plotMid2D' : True,
          'forcePP' : True,
          'plotCTScan' : True}

p = 912 # Index  
p = 11
p = 36 # Fails in filterBV on areas[-3]
patients = os.listdir(paRaw)

PP = ppV6(paRaw, paPP, params)
PP.process(patients[p])
    

#%% PPV6: Run for all patients

params['plotCTScan'] = False
params['plot3D'] = False
params['plotMid2D'] = False
params['force'] = False

patients = os.listdir(paRaw)
PP = ppV6(paRaw, paPP, params)
PP.process(patients)


#%% UNET: Set paths for predicting from PPV1 and ['UNETFromTutorial']
# ['UNETFromTutorial'] - > ['UNETPredPPV1_s2'] (512,512) 

allPaths = paths("S")

paMod = allPaths.paths['UNETFromTutorial']
if processStage==1:
    # Stage 1
    paPP = allPaths.paths['PPed']
    paPred = allPaths.paths['UNETPredPPV1']
elif processStage==2:
    # Stage 2
    paPP = allPaths.paths['PPed_s2']
    paPred = allPaths.paths['UNETPredPPV1_s2']


#%% Get UNET predictions - PPV1 and ['UNETFromTutorial']

files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

U = UNET(paMod, paPP, paPred, params=[], dimLim=[512,512]);
U.predictPPV1(files)


#%% UNET: Set paths for predicting from PPV1 and ['UNETTrained']
# This seems to be a 256x256 model

allPaths = paths("S")


paMod = allPaths.paths["UNETTrained"] #?
if processStage==1:
    # Stage 1
    paPP = allPaths.paths['PPed']
    paPred = allPaths.paths['UNETPredPPV1_UT']
elif processStage==2:
    # Stage 2
    paPP = allPaths.paths['PPed_s2']
    paPred = allPaths.paths['UNETPredPPV1_UT_s2']


#%% Get UNET predictions - PPV1 and ['UNETTrained']

files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

U = UNET(paMod, paPP, paPred, params=[], dimLim=[256,256]);
U.predictPPV1(files)


#%% UNET: Set paths for predicting from the other set...
# Also havd paMod = allPaths.paths["KerasUNETTest"], try that...
allPaths = paths("S")

paMod = allPaths.paths["KerasUNETTest"]
if processStage==1:
    # Stage 1
    paPP = allPaths.paths['PPed']
    paPred = allPaths.paths['UNETPredPPV1_UT2_s2']
elif processStage==2:
    # Stage 2
    paPP = allPaths.paths['PPed_s2']
    paPred = allPaths.paths['UNETPredPPV1_UT2_s2']
    

#%% Get UNET predictions - 'PPedStage2V2']

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

          
          
files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

U = UNET(paMod, paPP, paPred, params=[], dimLim=[256,256]);
U.predictPPV1(files)


#%% Generate basic features for XGB models
%load_ext autoreload
%autoreload 2

allPaths = paths()

# Use combined stage12 paths
ppPath1 = allPaths.paths['PPed_s12']
ppPath6 = allPaths.paths['PPedV6_s12']
nodesUNET = allPaths.paths['UNETPredPPV1_s12']
labelPath = allPaths.paths['labels_s12']

# Version 1
eF = features(paPPV1=ppPath1, paPPV6=ppPath6,
              labels=labelPath, nodesPath=None, name='S2V1') # Stage 1
eF.runV1(doTrain=1, doTest=0)
eF.runV1(doTrain=0, doTest=1)
# Version 3
eF2 = features(paPPV1=ppPath1, paPPV6=ppPath6,
              labels=labelPath, nodesPath=nodesUNET, name='S2V3') # Stage 1
eF2.runV3(doTrain=1, doTest=0)
eF2.runV3(doTrain=0, doTest=1)


#%% Train and predict xgb models
%load_ext autoreload
%autoreload 2

allPaths = paths()

# Train both models
mods = XGBMod(sFile, name='S2V1')
mods = mods.runAll()

# Train both models
mods = XGBMod(sFile, name='S2V3')
mods = mods.runAll()


#%% Train TF model


#%% Predict from TF model


#%% Ensemble selected models

%load_ext autoreload
%autoreload 2

sFile = allPaths.paths["SampleSub_s2"]

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
