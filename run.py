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
from ensemble import ens, ensTemplates

import os
from glob import glob
import pandas as pd

# Auto reload functions when testing
# %load_ext autoreload
# %autoreload 2

processStage = 2


#%% PPV1: Set paths 
""" 
- Set paths to use for pre-processing version 1
- Example processing and plotting for one patient (change p below)
- Batch processing all patients (no plotting)
"""

allPaths = paths()
if processStage==1:
    # Stage 1
    paRaw = allPaths.paths['Raw']
    paPP = allPaths.paths['PPed']
elif processStage==2:
    # Stage 2
    paRaw = allPaths.paths['Raw_s2']
    paPP = allPaths.paths['PPed_s2']

    
# 55 is upside down      
p = 912 # Index  
p = 1
patients = os.listdir(paRaw)

params = {'plotHist' : True,
          'plot3D' : True,
          'plotMid2D' : True,
          'forcePP' : True}

# Create object, process one patient
PP = ppV1(paRaw, paPP, params)
PP.process(patients[p])

# Loading example - now in ppV1.verify
ppV1.verify(paPP+patients[p]+'.npz')


# Process all patients
patients = os.listdir(paRaw)

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : True,
          'forcePP' : False}

# Create object, process all patients
PP = ppV1(paRaw, paPP, params)
PP.process(patients)


#%% PPV6: Set paths for ppV6
""" 
- Set paths to do pre-precoessing version 6
- Example processing and plotting for one patient (change p below)
- Batch processing all patients (no plotting)
"""  

allPaths = paths()
if processStage==1:
    # Stage 1
    paRaw = allPaths.paths['Raw']
    paPP = allPaths.paths['PPedV6']
elif processStage==2:
    # Stage 2
    paRaw = allPaths.paths['Raw_s2']
    paPP = allPaths.paths['PPedV6_s2']


# PPV6: Run for one patient
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
    
# PPV6: Run for all patients
params['plotCTScan'] = False
params['plot3D'] = False
params['plotMid2D'] = False
params['force'] = False

patients = os.listdir(paRaw)
PP = ppV6(paRaw, paPP, params)
PP.process(patients)


#%% Train UNET
""" 
Train UNET model using LUNA16 dataset
TODO:
    - Add tidied training and proprocessing code to UNET class
    - Add code for training the two models used below
"""


#%% UNET: Prdict from PPV1 and ['UNETFromTutorial']
"""
Predict from the UNET model based on the Kaggle tutorial
Save nodule predictions.

 ['UNETFromTutorial'] - > ['UNETPredPPV1_s2'] (512,512) 
"""

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


files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

U = UNET(paMod, paPP, paPred, params=[], dimLim=[512,512]);
U.predictPPV1(files)


#%% UNET: Predict from PPV1 and ['UNETTrained']
"""
Preict from a modifided verison of the UNET model. Smaller image size, LUNA16
data preprocessed differentely from Kaggle Tutorial verion.
"""

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


# Get UNET predictions - PPV1 and ['UNETTrained']

files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

U = UNET(paMod, paPP, paPred, params=[], dimLim=[256,256]);
U.predictPPV1(files)


#%% Generate basic features for XGB models
"""
Extracts various features from 3D scans.
Air volume/ratios, blood volume/ratios, malignant nodule value rations, etc.
Inlcudes early version (V1) and later verision (V3). V3 generates more 
feratures (some redundent).

Then fits XGBTree and XGBDart models to these features (XGBMod class handles 
this).
"""

allPaths = paths()

# Use combined stage12 paths
ppPath1 = allPaths.paths['PPed_s12']
ppPath6 = allPaths.paths['PPedV6_s12']
nodesUNET = allPaths.paths['UNETPredPPV1_s12']
labelPath = allPaths.paths['labels_s12']

# Version 1
eF = features(paPPV1=ppPath1, paPPV6=ppPath6,
              labels=labelPath, nodesPath=None, name='S2V1') # Stage 1
eF.runV1(doTrain=1, doTest=0) # Generate for training set
eF.runV1(doTrain=0, doTest=1) # Generate for test set

# Version 3
eF2 = features(paPPV1=ppPath1, paPPV6=ppPath6,
              labels=labelPath, nodesPath=nodesUNET, name='S2V3') # Stage 1
eF2.runV3(doTrain=1, doTest=0) # Generate for training set
eF2.runV3(doTrain=0, doTest=1) # Generate for test set


# Train and predict xgb models

# Train both models
mods = XGBMod(sFile, name='S2V1')
mods = mods.runAll()

# Train both models
mods = XGBMod(sFile, name='S2V3')
mods = mods.runAll()


#%% Train TF models
"""
Trains 3D convolution neural network on extracted nodules
TODO:
    - Add training code to tfMod class
    - For UNET nodes and for "Basic nodes"
"""

#%% Predict from TF model
"""
Preidct from the two TF models (one ttrained on UNEt nodules, other on basic
nodules)
TODO:
    - Preidiction code already in tfMod. Add here.
"""

#%% Ensemble selected models
"""
Create enseimbles of various models using XGBoost (handled in ens class).

Loads predictions from models from prediction folder.
Various templates are held in ensTemplates.
Two methods:
- XGBTree - if training predictions are available, use to fit another
 XGB Model with them
- Mean - Take simple mean of predicitions. Works without training predicitions,
 can be weighted
 
TODO:
    - Update mean method so that it works without training predicitons when 
    weights are suppliied.

"""

sFile = allPaths.paths["SampleSub_s2"]


#%% XGB_V1
ens1 = ensTemplates.XGBs_V1(sFile)


#%% XGB_V2
ens2 = ensTemplates.XGBs_V2(sFile)


#%% XGB_V3
ens3 = ensTemplates.XGBs_V3(sFile)


#%% XGB_V4 and 5
ens4 = ensTemplates.XGBs_V4(sFile)
ens5 = ensTemplates.XGBs_V5(sFile)


#%% XGB_V7
ens7 = ensTemplates.XGBs_V7(sFile)


#%% XGB_V8
ens8 = ensTemplates.XGBs_V8(sFile)


#%% XGB_V9
ens9 = ensTemplates.XGBs_V9(sFile)
