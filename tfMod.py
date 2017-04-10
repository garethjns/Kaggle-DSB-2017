# -*- coding: utf-8 -*-
"""
TFMod class (from trainV3 and predictV3)

Handles training and predicting from node model in TF

tfHelpers inherits fHelpers from generateFeatures which includes basic
node extraction methods. Need to make sure these are the same versions, if not,
overload with version from trainV3 script. 

getXY looks same
getCols looks same
getZ looks same
getData is different version!
extractNodes looks same
extractNodes2 looks same
"""

#%% Imports - incomplete

from preprocessing import plot3D, plot_3d
from generateFeatures import fHelpers

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import random as rand


#%% Helpers
# A few functions that do the remaning PP steps to the data after loading
# from disk

class tfHelpers(fHelpers):
    @staticmethod
    def normalize(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image.astype(np.float16)
        
    @staticmethod
    def zero_center(image, PIXEL_MEAN = 0.25):
        image = image - PIXEL_MEAN
        return image.astype(np.float16)

    
    # Overload fHelpers.getData with this version from trainV3.
    @staticmethod
    def getData(data, xyzMin, xyzMax, buffs=[20,20,10], plotOn=False):
    
        # First make sure xyzMax is inside data.shape
        # It might not be if coords come from mask larger than data
        # As xyzMin is rescaled, then xyzMax calced as xyzMIn=buffs*2
        # This grabs more space in real terms
        # No check is made on size when this is done
        
        # Check max of each column
        bigIdx = xyzMax > data.shape[::-1]
        # print(bigIdx)
        # Remove rows where any too big
        rmIdx = np.any(bigIdx, axis = 1)
        # Any drop these rows
        xyzMin = xyzMin[rmIdx==False,:]
        xyzMax = xyzMax[rmIdx==False,:]
        # Warns if no valid rows remaining
        
        # Note buff sized used here must be specified (no data to calculate)
        if len(xyzMin) == 0:
            print('No data available')
            # For now, if now candidates are available, return 1 cube of zeros 
            # for this patient
            out = np.zeros(shape=[1,buffs[2]*2,buffs[1]*2,buffs[0]*2], dtype=np.int16)
          
            return out, 0
            
        else:
            # Note buff size is recalcualted here. Specified is ignored.
            n = len(xyzMin)
            xSize = int(xyzMax[0,0] - xyzMin[0,0])
            ySize = int(xyzMax[0,1] - xyzMin[0,1])
            zSize = int(xyzMax[0,2] - xyzMin[0,2])
            
            out = np.zeros(shape=[n,zSize,ySize,xSize], dtype=np.int16)
            for ni in range(0,n):
                # print(ni)
                # Data is zyx
                # out is zyx
                out[ni,:,:,:] = data[\
                 xyzMin[ni,2]:xyzMax[ni,2], \
                 xyzMin[ni,1]:xyzMax[ni,1], \
                 xyzMin[ni,0]:xyzMax[ni,0]]
                                
                if plotOn:
                    plot_3d(out[ni,:,:,:], 0)
                    plt.show()
            
            return out, n  
            
    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

           
#%% TF Mod
""" 
Class handelling the tensorFlow model

.buildTrainingNetwork builds network for training
.runTraining runs the taining
.builtTestNetwork (to do) builds same network for predicting, but different placeholders
.runTest() will be from predict script (not added yet)  
"""

class tfMod(tfHelpers):
    def __init__(self, modPath, dataPath, nodePath, labelPath, cParams, 
                 nodeMode=1, nMax=50,
                 num_channels=1, num_labels=2, validProp=0.15, 
                 processingStage=1, train=1, batch_size=50, num_hidden=560,
                 patch_size=5, outChans=80):
        
        self.modPath = modPath
        self.dataPath = dataPath
        self.labelPath = labelPath
        
        # Set model params
        self.batch_size = batch_size
        self.nBatch = batch_size
        self.patch_size = patch_size
        self.num_hidden = num_hidden
        self.outChans = outChans
        self.num_steps = 60000 # training steps
        
        # Set data params
        self.zSize = cParams['zBuff']*2
        self.ySize = cParams['yBuff']*2 
        self.xSize = cParams['xBuff']*2
        self.num_channels = num_channels
        self.train = train
        self.nodeMode = nodeMode
        self.nodePath = nodePath
        
        if train==1:
            # Set up for training
            if processingStage==1:
                
                allFiles, trainFiles, testFiles, _, nTrainFiles, _, labelsCSV = \
                self.availFilesStage1(self.dataPath, self.labelPath)
                
                nLoad = nTrainFiles
                loadIdx = np.random.choice(range(0,trainFiles.shape[0]), nLoad, 
                                                 replace=False)
                splitLoad = round(nLoad*(0-validProp))
                trainIdx = loadIdx[0:splitLoad]
                validIdx = loadIdx[splitLoad:nLoad]
                
                trainFilesLoad = trainFiles.iloc[trainIdx]                                   
                trainLabelsLoad = labelsCSV.iloc[trainIdx]
                cTrain, cTrainLabels = self.loadPPFilesV7(dataPath,
                                      trainFilesLoad, cParams, trainLabelsLoad, 
                                      nodeMode=2)
                
                validFilesLoad = trainFiles.iloc[validIdx]                                   
                validLabelsLoad = labelsCSV.iloc[validIdx]
                cValid, cValidLabels = self.loadPPFilesV7(dataPath,
                                      validFilesLoad, cParams, validLabelsLoad, 
                                      nodeMode=2)
            
            elif processingStage==2:
                # TODO:
                # Add: Retrain on all data
                # Add: Get test files from new samplesub
                pass
            
            self.allFiles = allFiles
            self.trainFiles = trainFiles
            self.nTrainFiles = nTrainFiles
            self.nLoad = nLoad
            self.trainFilesLoad = trainFilesLoad
            self.trainLabelsLoad = trainLabelsLoad
            self.cTrain = cTrain
            self.cTrainLabels = cTrainLabels
            self.validFilesLoad = validFilesLoad
            self.validLabelsLoad = validLabelsLoad
            self.cValid = cValid
            self.cValidLabels = cValidLabels
            self.nVaid = 200
        else:
            # Set up for testing
            # Load test files, change model placeholders
            self.nTest = 1216#testFiles.shape[0]
            
    @staticmethod
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])    
   
    @staticmethod        
    def availFilesStage1(path, labelPath):
        # Find preprocess files
        availFiles = pd.DataFrame(os.listdir(path))
        nAvailFiles = availFiles.shape[0]
        # Find those that are in the training set
        labelsCSV = pd.read_csv(labelPath)
        
        a = np.empty([nAvailFiles,1])
        for r in range(0,nAvailFiles):
            
            fn = availFiles.iloc[r,0][0:-4]
        
            a[r,0] = (fn in set(labelsCSV['id']))
            
            #print(fn in set(labelsCSV['id']))
            
      
        nAvailTrain = int(sum(a))
        nAvailTest = int(sum(a==0))
      
        availTrain = availFiles[a==1]
        availTest = availFiles[a==0]
        
        return(availFiles, availTrain, availTest, 
               nAvailFiles, nAvailTrain, nAvailTest)    
    
    # Load file, extract candidates, discard rest of file. Return candidates
    def loadPPFilesV6(self, path, files, cParams, labels):
        
        if isinstance(files, str):
            files = [files]
        elif isinstance(files, pd.DataFrame):
            files = set(files.iloc[:,0])
        else:
            files = set(files)
            
           
        nFiles = len(files)
        # loaded = []
        
        xBuff = cParams['xBuff'] # in each direction (size=*2)
        yBuff = cParams['yBuff'] # in each direction
        zBuff = cParams['zBuff'] # in each direction
        nMax = cParams['nMax']
        
        loaded = np.empty(shape=[0,zBuff*2,yBuff*2,xBuff*2,1])
        newLabels = pd.DataFrame(columns=['id', 'cancer'])  
        
        r = 0     
        for fn in files:
            print('Attempting to load: ' + path+fn)    
    
            data = np.load(path+fn)['arr_0']
    
            # c, cLabs = getNodesAndLabels([data], \
            #                    cParams=cParams, labels=labels.iloc[r,:])
    
            
            # Get candidaes
            smData, n, xyzMin, xyzMax = self.extractNodes(
                                data, buffs=[xBuff,yBuff,zBuff], 
                                nMax = nMax, plotOn=False)
            
            
            if n == 0:
                # n returned is zero if data is empty
                n = 1
                
            # Append candidates to output
            loaded = np.append(loaded, np.expand_dims(smData, axis=4), axis=0)
            newLabels = newLabels.append([labels.iloc[r,:]]*n)
            
            r +=1
            print('Added: ' + fn + ' (' + str(r) + '/' + str(nFiles) + ')')
        return loaded, newLabels
            
    # In this version load mask, then get data from mask or non-mask data 
    # Give paths, so hardcode choices for now
    def loadPPFilesV7(self, files):
        
        if isinstance(files, str):
            files = [files]
        elif isinstance(files, pd.DataFrame):
            files = set(files.iloc[:,0])
        else:
            files = set(files)
    
        nFiles = len(files)
        
        xBuff = self.xBuff # in each direction (size=*2)
        yBuff = self.yBuff # in each direction
        zBuff = self.zBuff # in each direction
        nMax = self.nMax
        
        loaded = np.empty(shape=[0,zBuff*2,yBuff*2,xBuff*2,1])
        newLabels = pd.DataFrame(columns=['id', 'cancer'])  
        
        # Needs updating
        path2 = self.dataPath # PPV1
        path1 = self.nodePath # UNET or PPV6 nodes - should correspond with nodeMode
        path3 = self.nodePath
        
        r = 0     
        for fn in files:
            print('Attempting to load data: ' + path1+fn )    
            
            data2 = np.load(path2+fn)['resImage3D']

            if self.nodeMode == 1:
                # Use simple nodes
                print('Attempting to load nodes: ' + path1+fn)
                data1 = np.load(path1+fn)['arr_0']
                
            elif self.nodeMode == 2:
                print('Attempting to load nodes: ' + path3+fn)
                # Use UNET nodes
                data1 = np.load(path3+fn)['nodPreds'].astype(np.float32)
                data1 = np.squeeze(data1)
                print(np.sum(data1))
    
    
            # c, cLabs = getNodesAndLabels([data], \
            #                    cParams=cParams, labels=labels.iloc[r,:])
    
            # Get candidaes
            smData, n, xyzMin, xyzMax = self.extractNodes2(
                                data1=data1, data2=data2, buffs=[xBuff,yBuff,zBuff], 
                                nMax = nMax, plotOn=False)
            
            if n == 0:
                # n returned is zero if data is empty
                n = 1
                
            # Append candidates to output
            self.loaded = np.append(self.loaded, np.expand_dims(smData, axis=4), axis=0)
            self.newLabels = newLabels.append([self.labels.iloc[r,:]]*n)
            
            r +=1
            print('Added: ' + fn + ' (' + str(r) + '/' + str(nFiles) + ')')
        
        return loaded, newLabels
        
    @staticmethod    
    def getNodesAndLabels(data, cParams, labels):
        # Gets nodes and labels in usable formats
        # Expand out extra dim on loaded (to work with 3D conv)
        
        # Data is a list now
        
        # Set params (update to from input)    
        xBuff = cParams['xBuff'] # in each direction (size=*2)
        yBuff = cParams['yBuff'] # in each direction
        zBuff = cParams['zBuff'] # in each direction
        nMax = cParams['nMax']
    
        # Create empty array to append data to     
        loaded = np.empty(shape=[0,zBuff*2,yBuff*2,xBuff*2,1])
        # Create empty df to append labels to
        labels = pd.DataFrame(columns=['id', 'cancer'])
        
        # For each file
        nData = len(data)
        r=-1
        for d in data:
            r+=1
         
            # Get candidaes
            smData, n, xyzMin, xyzMax = tfMod.extractNodes(
                                d, buffs=[xBuff,yBuff,zBuff], 
                                nMax = nMax, plotOn=False)
            # Append candidates to output
            loaded = np.append(loaded, np.expand_dims(smData, axis=4), axis=0)
            # Append n labels to df
            if n == 0:
                # n returned is zero if data is empty
                n = 1
                
            labels = labels.append([labels]*n)
            
            print('Done ' + str(r) + '/' + str(nData))
            
        return loaded, labels
        
    def buildTrainingNetwork(self):
        # Build network 
        # Params set to self in init.
        # 2 conv layers, 1 fully connected layer
        # this is n rows expected by placeholder - files produce multiple rows
        # nBatch = batch_size
        # depth = 1
        # nTest = 1216#testFiles.shape[0]
        # nTrain = round(nBatch*(1-validProp))
        # nValid = cValid.shape[0]*0.5
        # nValid = 200
        self.graph = tf.Graph()
        
        with graph.as_default():
        
          # Input data.
          trainSubset = tf.placeholder(tf.float32, shape=(self.nTrain, 
                               self.zSize, self.ySize, self.xSize, self.num_channels), name='PHTrain')
          trainSubsetLabels = tf.placeholder(tf.float32, shape=(self.nTrain,2), name='PHTrainLabs')
          
          validSubset = tf.placeholder(tf.float32, shape=(self.nValid, 
                               self.zSize, self.ySize, self.xSize, self.num_channels), name='PHValid')
          validSubsetLabels = tf.placeholder(tf.float32, shape=(self.nValid,2), name='PHValidLabs')
          
          testDataset = tf.placeholder(tf.float32, shape = (self.nTest, 
                                                 self.zSize, self.ySize, self.xSize, self.num_channels), name='PHTest')
          
          # testDataset = tf.Variable(initial_value = [1.0,1.0,1.0,1.0,1.0], validate_shape=False, name='PHTest')
          LR = tf.Variable(1)
          # testDataset = tf.constant(testData)
          
          outChans = self.outChans
          
          # Variables.
          layer1_weights = tf.Variable(tf.truncated_normal(
              [self.patch_size, self.patch_size, self.patch_size, self.num_channels, outChans], stddev=0.1))
        
          layer1_biases = tf.Variable(tf.zeros([outChans]))
          
          layer2_weights = tf.Variable(tf.truncated_normal(
              [self.patch_size, self.patch_size, self.patch_size, outChans, outChans], stddev=0.1))
        
          layer2_biases = tf.Variable(tf.constant(1.0, shape=[outChans]))
          
          # layer2b_weights = tf.Variable(tf.truncated_normal(
          #    [patch_size, patch_size, patch_size, outChans, outChans], stddev=0.1))
        
          # layer2b_biases = tf.Variable(tf.constant(1.0, shape=[outChans]))
        
          layer3_weights = tf.Variable(tf.truncated_normal(
              [self.xSize // 4 * self.ySize // 4 * self.zSize //4 * outChans, 
               self.num_hidden], stddev=0.1))
        
          layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
        
          layer4_weights = tf.Variable(tf.truncated_normal(
              [self.num_hidden, self.num_labels], stddev=0.1))
        
          layer4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))
          
          # Model.
          def model(smData):
            # conv = tf.nn.conv3d(data.astype(np.float32), layer1_weights, [1, 2, 2, 2, 1], padding='SAME')
            conv = tf.nn.conv3d(smData, layer1_weights, [1, 2, 2, 2, 1], padding='SAME')
            # conv = tf.nn.max_pool3d(smData, [1,2,3,4,5], [1,2,3,4,5], padding='SAME')
            hidden = tf.nn.tanh(conv + layer1_biases)
            
            conv = tf.nn.conv3d(hidden, layer2_weights, [1, 2, 2, 2, 1], padding='SAME')
            hidden = tf.nn.dropout(conv + layer2_biases, 0.8)
            
            # conv = tf.nn.conv3d(hidden, layer2b_weights, [1, 2, 2, 2, 1], padding='SAME')
            # hidden = tf.nn.tanh(conv + layer2b_biases)
            
            # shape = hidden.get_shape().as_list()
            shape = hidden.get_shape().as_list()
            # sh ape = [2, 5, 10, 10, 50]
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3] * shape[4]])
        
            hidden = tf.nn.dropout(tf.matmul(reshape, layer3_weights) + layer3_biases, 0.8)
        
            return tf.matmul(hidden, layer4_weights) + layer4_biases
          
          # Training computation.
          self.logits = model(trainSubset) 
          self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=trainSubsetLabels, 
                                                    logits=self.logits))
          
          self.logitsV = model(validSubset)
          self.lossV = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=validSubsetLabels, 
                                                    logits=self.logitsV))
          
          # Optimizer.
          self. optimizer = tf.train.GradientDescentOptimizer(LR).minimize(self.loss)
          
          # Predictions for the training, validation, and test data.
          self.train_prediction = tf.nn.softmax(self.logits)
          self.valid_prediction = tf.nn.softmax(self.logitsV)
          
          self.test_prediction = model(testDataset)
          self.test_prediction = tf.nn.softmax(model(testDataset))
          
          # Add ops to save and restore all the variables.
          self.saver = tf.train.Saver()
          
    def runTraining(self):
        # Run training - load all first
        
        initialRate = 0.000001
        initialRate = 0.001
        
        scalingFactor = 6
        
        rand.seed(123123)
        
        self.tsAcc = []
        self.vsAcc = []
        self.tLoss = []
        self.vLoss = []
        it = -1
        self.bestPerf = 0
        sinceLast = 0
        with tf.Session(graph=self.graph, 
                config=tf.ConfigProto(log_device_placement=True)) as session:
            
            tf.global_variables_initializer().run()
            print('Initialized')
            #print(session.run(logits))
            for step in range(self.num_steps):
                it+=1       
               
                learningRate = initialRate/np.exp(it/(self.num_steps/scalingFactor))        
                
                # Get data and labels
                # allData -> trainLoad -> this batch
                fIdxTrain = np.random.choice(range(0, cTrain.shape[0]), 
                                                   round(self.nBatch*(1-self.validProp)), 
                                          replace=False)
                fIdxValid = np.random.choice(range(0, self.cValid.shape[0]), 
                                                   round(self.nValid),
                                          replace=False)
        
                
                # New: Data set already loaded
                cTrainSubset = self.cTrain[fIdxTrain,:,:,:,:]
                cTrainLabelsSubset = self.cTrainLabels['cancer'].iloc[fIdxTrain]
                cValidSubset = self.cValid[fIdxValid,:,:,:,:]
                cValidLabelsSubset = self.cValidLabels['cancer'].iloc[fIdxValid]

              
                feed_dict = {self.trainSubset : cTrainSubset, 
                             self.trainSubsetLabels : self.oneHot(cTrainLabelsSubset,2),
                             self. validSubset : cValidSubset, 
                             self.validSubsetLabels : self.oneHot(cValidLabelsSubset,2),
                             self.LR : learningRate}
                             # testDataset : testData}
                             
                _, l, predictions, lv, lp = session.run(
                [self.optimizer, self.loss, self.train_prediction, self.lossV, self.valid_prediction], feed_dict=feed_dict)
                
                # if (step % 50 == 0):
                print('-------------------------------------------------------------')
                print('LR: ' + str(learningRate))
                print('Minibatch loss at step %d: %f' % (step, l))
                
                self.tsAcc.append(self.accuracy(predictions, self.oneHot(cTrainLabelsSubset)))
                self.tLoss.append(l)
                print("Training accuracy: " + str(self.tsAcc[it]))
                
                # lv, lp = valid_prediction.eval(feed_dict=feed_dict)
                
                self.vsAcc.append(self.accuracy(lp,self.oneHot(cValidLabelsSubset)))
                vLoss.append(lv)
                
                print("Validation accuracy: " + str(self.vsAcc[it]))
                print("Validation loss: " + str(lv))
                
                
                print('Training acc:')
                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                
                plt.plot(self.tsAcc)
                plt.plot(self.vsAcc, c='r')
                # plt.title('A tale of 2 subplots')
                plt.ylabel('Accuracy')
                plt.xlabel('Step')
                # plt.show()
                
                
                print('Training loss:')
                plt.subplot(1,2,2)
                l1 = plt.plot(self.tLoss, label='Train')
                l2 = plt.plot(self.vLoss, c='r', label='Valid')
                plt.legend(labels=['Train', 'Valid'])
                plt.ylabel('Loss')
                plt.xlabel('Step')
                
                plt.show()

                # Save the variables to disk.
                # Rolling average of 5 steps
                sinceLast +=1
                if step > 5:
                    # Current perf is this step and previous 4
                    currentPerf = np.mean(self.vsAcc[it-5:it])
                    # If this is better than the bestPerf, update bestPerf and 
                    # checkpoint model
                    if currentPerf >= self.bestPerf and sinceLast>10:
                        sinceLast = 0
                        self.bestPerf=currentPerf
                        chkFn = "S:\OneDrive\Matlab\DSB2017\Kaggle-DSB2017\\tmp\\model_st" + str(step) + '_' + \
                        str(round(float(currentPerf),3)) + ".ckpt"
        
                        save_path = self.saver.save(session, chkFn)
                        
                        print("Model saved in file: %s" % save_path)          
               
            # Save the variables to disk.
            save_path = saver.save(session, paths['TF3D_40'])
            print("Model saved in file: %s" % save_path)   
             
        print('Training acc:')
        plt.plot(tsAcc)
        plt.show()
        print('Validation acc:')
        plt.plot(vsAcc)
        plt.show()
