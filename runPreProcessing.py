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
import os

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
params['plot3D'] = True
params['plotMid2D'] = False
params['force'] = True

patients = os.listdir(paRaw)
PP = ppV6(paRaw, paPP, params)
PP.process(patients)


#%% UNET: Set paths for predicting from PPV1

allPaths = paths()

paMod = []
if processStage==1:
    # Stage 1
    paPP = allPaths.paths['PPed']
    paPred = []
elif processStage==2:
    # Stage 2
    paPP = allPaths.paths['PPed_s2']
    paPred = []


#%% Get UNET predictions - PPV1

files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

U = UNET(paMod, paPP, paPred);
U.predictPPV1(files, params, dimLim=[512,512])


#%% UNET: Set paths for predicting from the other set...

allPaths = paths()

paMod = []
if processStage==1:
    # Stage 1
    paPP = paths['PPedStage2V2']
    paPred = []
elif processStage==2:
    # Stage 2
    paPP = paths['PPedStage2V2_s2']
    paPred = []
    

#%% Get UNET predictions - 'PPedStage2V2']

params = {'plotHist' : False,
          'plot3D' : False,
          'plotMid2D' : False,
          'forcePP' : False,
          'forcePred' : False}

          
files = glob(paPP+'*.npz')
files = pd.DataFrame({'Files' : files})

U = UNET(paMod, paPP, paPred);
U.predictPPStage2V2(files, params, dimLim=[512,512])


#%% Generate basic features for XGB models


#%% Predict from XBG model


#%% Train TF model


#%% Predict from TF model


#%% Ensemble selected models
