# -*- coding: utf-8 -*-
"""
UNET Handler
Adding prediction code first, as needed for stage 2
- Adding two predictions methods used for stage 1
- UNETPPV1 (UNETPredictPPV1, now predictPPV1)
- And UNETPPS2V2 (originally UNETPredictPPStage2V2, now predictPPStage2V2)


TO DO:
    - Add LUNA16 preprocessing code
    - Add UNET training code
"""

#%% Imports
import numpy as np

import os
import matplotlib.pyplot as plt

from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf 
from keras.models import load_model



#%% uHelpers
# Containing prcoessing methods

class uHelpers():
    @staticmethod
    def dice_coef(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return -uHelpers.dice_coef(y_true, y_pred)
        
    @staticmethod    
    def placeInDims3D(data3D, dims=[256,256]):
    
        sh = data3D.shape
        
        loaded = np.zeros([sh[0], dims[0], dims[1]], dtype = np.float32)-1024
    
        axIn = list()
        axOut = list()
        # Lists[z,x,y], where each = list of start, end 
        for ax in range(0,2):
            axLen = data3D.shape[ax+1]
            if axLen == dims[ax]:
                # Same size
                axIn.append([0, dims[0]])
                axOut.append([0, dims[0]]) 
            if  axLen < dims[ax]:
               # File data smaller than output - buffer
               axDiff = round((dims[ax] - axLen) / 2)
             
               axIn.append([0, axLen])
               axOut.append([axDiff, axDiff+axLen])
           
            if axLen > dims[ax]:
            # File data larger than output - crop from middle 
               axDiff = (dims[ax] - axLen) / 2
               # Note - rounding is weird
               axIn.append([abs(round(axDiff)), round(axLen+axDiff)])
               axOut.append([0, dims[ax]])
           
        # Place in output
        loaded[:, axOut[0][0]:axOut[0][1], \
          axOut[0][0]:axOut[0][1]] = \
        data3D[:, axIn[0][0]:axIn[0][1], \
          axIn[0][0]:axIn[0][1]]
              
        return loaded
        
    @staticmethod
    def norm(data3D):
        
        ot = data3D.dtype
        if ot != np.float64:
        # Low percision risks nan std
            data3D = data3D.astype(np.float64)
    
        mean = np.mean(data3D)  # mean for data centering
        std = np.std(data3D)  # std for data normalization
    
        data3D -= mean
        data3D /= std
        
        return data3D.astype(ot)
    @staticmethod    
    def normMM(d):
        return (d-np.min(d)) / (np.max(d)-np.min(d))
    
    @staticmethod    
    def verifyUNET(fn):
        
        try:
            l = np.load(fn)    
            nodPreds = l['nodPreds']
            print(np.sum(nodPreds))
            print('Verified.')
            return(True)
        except:
            print('Failed.')
            return(False)
            

#%% UNET
# Containing training and predicting methods

class UNET(uHelpers):
    def __init__(self, modPath, ppPath, predPath, params, dimLim=[256,256]):
        self.modPath = modPath # eg. ['UNETFromTutorial']
        self.ppPath = ppPath # eg. for predictPPV1 ['PPedSSD']
        self.predPath = predPath # ['UNETPredPPV1']
        self.dimLim = dimLim
        
        self = self.loadMod()
        
    
    def loadMod(self):
        self.UNETMod = load_model(self.modPath, 
                  custom_objects={'dice_coef': uHelpers.dice_coef, 
                  'dice_coef_loss' : uHelpers.dice_coef_loss})       
    
    def train(self):
        pass
    
    def predictPPV1(self, files):
        # Load from ['PPedSSD'], save to ['UNETPredPPV1']
        
        #if isinstance(files, pd.DataFrame):
        #    files = set(files['Files'])
    
        nFiles = len(files)
    
        for f in range(0, nFiles):
            
            fn = files.iloc[f][0]
            do = True
            print('Loading: ' + fn + ' : ' + str(f+1) + '/' + str(nFiles))
            
            fnSave = self.predPath+fn[len(self.ppPath)::]
            if os.path.isfile(fnSave) and not self.params['forcePred']:   
                print('   Already done.')
            # But only skip if ok
                if not self.verifyUNET(fnSave):
                    do = True
                else:
                    do = False
    
            if do:
                l = np.load(fn)    
                resImage3D = l['resImage3D']
                segmented_lungs_fill = self.norm(l['segmented_lungs_fill'])
                
                # Apply lung mask
                resImage3D[segmented_lungs_fill==0] = -1024
                
                # Place and expand
                placed = self.placeInDims3D(resImage3D, dims=self.dimLim)
                placed = np.expand_dims(placed, axis=1)
                            
                # Normalise in the same way as UNET train set
                placed = self.norm(placed)
                preds = self.UNETMod.predict(placed, batch_size = 2, verbose=1)
                
                print('Saving: ' + fnSave)
                np.savez_compressed(fnSave, nodPreds=preds.astype(np.float16))
    
                print('Sum: ' + str(np.sum(preds)))
                print('Max: ' + str(np.max(preds)))
                print('Mean: ' + str(np.mean(preds)))
                print('Med: ' + str(np.median(preds)))
                
                preds = self.normMM(preds.squeeze())
                plt.figure(figsize =(16,4))
                plt.subplot(1,3,1)
                plt.imshow(self.norm(preds.sum(axis=0)))
                plt.subplot(1,3,2)
                plt.imshow(preds.sum(axis=1))
                plt.subplot(1,3,3)
                plt.imshow(preds.sum(axis=2))
                plt.show()
                
                placed = placed.squeeze()
                plt.figure(figsize =(16,4))
                plt.subplot(1,3,1)
                plt.imshow(placed.sum(axis=0))
                plt.subplot(1,3,2)
                plt.imshow(placed.sum(axis=1))
                plt.subplot(1,3,3)
                plt.imshow(placed.sum(axis=2))
                plt.show()
                
                
    def predictPPStage2V2(self, paths, files, params, dimLim=[256,256]):
        # Load from ['PPedStage2V2'], save to ['UNETPredPPS2V2']
    
        #if isinstance(files, pd.DataFrame):
        #    files = set(files['Files'])
    
        nFiles = len(files)
    
        for f in range(0, nFiles):
            
            fn = files.iloc[f][0]
            do = True
            print('Loading: ' + fn + ' : ' + str(f+1) + '/' + str(nFiles))
            
          
            fnSave = paths['UNETPredPPS2V2']+fn[len(paths['PPedStage2V2'])::]
            if os.path.isfile(fnSave) and not params['forcePred']:   
                print('   Already done.')
            # But only skip if ok
                if not self.verifyUNET(fnSave):
                    do = True
                else:
                    do = False
    
            if do:
                l = np.load(fn)    
                # resImage3D = l['resImage3D']
                # segmented_lungs_fill = norm(l['segmented_lungs_fill'])
                
                # Apply lung mask
                # resImage3D[segmented_lungs_fill==0] = -1024
                
                resImage3D = l['image3DMasked']
    
                # Place and expand
                placed = self.placeInDims3D(resImage3D, dims=dimLim)
                placed = np.expand_dims(placed, axis=1)
                            
                # Normalise in the same way as UNET train set
                placed = self.norm(placed)
                preds = UNET.predict(placed, batch_size = 2, verbose=1)
                
                np.savez_compressed(fnSave, nodPreds=preds.astype(np.float16))
    
                print('Sum: ' + str(np.sum(preds)))
                print('Max: ' + str(np.max(preds)))
                print('Mean: ' + str(np.mean(preds)))
                print('Med: ' + str(np.median(preds)))
                
                preds = self.normMM(preds.squeeze())
                plt.figure(figsize =(16,4))
                plt.subplot(1,3,1)
                plt.imshow(self.norm(preds.sum(axis=0)))
                plt.subplot(1,3,2)
                plt.imshow(preds.sum(axis=1))
                plt.subplot(1,3,3)
                plt.imshow(preds.sum(axis=2))
                plt.show()
                
                placed = placed.squeeze()
                plt.figure(figsize =(16,4))
                plt.subplot(1,3,1)
                plt.imshow(placed.sum(axis=0))
                plt.subplot(1,3,2)
                plt.imshow(placed.sum(axis=1))
                plt.subplot(1,3,3)
                plt.imshow(placed.sum(axis=2))
                plt.show()
    
    
    params = {'plotHist' : False,
              'plot3D' : False,
              'plotMid2D' : False,
              'forcePP' : False,
              'forcePred' : False}
              