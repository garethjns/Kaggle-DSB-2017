# -*- coding: utf-8 -*-
"""
Train a basic 3D convolutional neural network on 3D area isolated around node
locations.
Works with TensorBoard
Structure eg.:
3D Conv -> tanh -> 3D Conv -> dropout -> FC layer 

Uses either node locations predicted from UNET or "basic nodes". Uses the basic
node finding code in fHelpers (extractFeatures.py), with a couple of 
modifications. Modified funtions are overloaded in tfHelpers here.

tfMod.buildNetwork() will build network for training and predicting, difference
is size of placeholders. In Training train and valid data placeholders are ~300
and ~200. For testing only 1 is created with shape[0] = 50.

TODO:
    - Most training and test code have been added, but won't work yet. 
    Need slight modification as now inside functions instead of script
    - Need to update parameter setting
    - Function calls to appropriate objects
    - Need to add lines to run to run.py
"""

#%% Imports
# fHelpers already imports pHelpers
from extractFeature import fHelpers

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# In Get GPUs function:
# from tensorflow.python.client import device_lib


#%% Training/predicting methods

class tfHelpers(fHelpers):
    """ 
    Based on Train/predict V3_TB
    Adding functions in helpers below
    Adding test code as functions here - not creating an object
    """
    def normalize(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image.astype(np.float16)
        

    def zero_center(image, PIXEL_MEAN = 0.25):
        image = image - PIXEL_MEAN
        return image.astype(np.float16)
        
    def oneHot(labels, cats=0):
        labels = np.array(labels).astype(int)
        
        n = len(labels)
        nCat = len(np.unique(labels))
        # If cats is specified, check all are represeted in this set    
        if nCat<cats:
            nCat = cats
        
        hotLabels = np.zeros([n, nCat])   
        
        for r in range(0,n):
            hotLabels[r,labels[r]-1] = 1
        
        return(hotLabels)
        
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
        
    
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
      
    def getData(data, xyzMin, xyzMax, buffs=[20,20,10], plotOn=False):
        """
        This version of getData is from trainV3_TB (TF mod) training and
        replaces the older (?) version in extractFeatures.fHelpers
        
        First make sure xyzMax is inside data.shape
        It might not be if coords come from mask larger than data
        As xyzMin is rescaled, then xyzMax calced as xyzMIn=buffs*2
        This grabs more space in real terms
        No check is made on size when this is done
        """
        
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
                    tfHelpers.plot_3d(out[ni,:,:,:], 0)
                    plt.show()
            
            return out, n  
            
    def getNodesAndLabels(data, cParams, labels):
        """
        Gets nodes and labels in usable formats
        Expand out extra dim on loaded (to work with 3D conv)
        
        Data is a list now
        """
        
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
            smData, n, xyzMin, xyzMax = tfHelpers.extractNodes(
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
        
        
    def reducePreds(preds, cTestLabels):

        cTestLabels.cancer = preds[:,0]
        
        predsRed = cTestLabels.groupby(['id'], as_index=False).mean()
        
        predsRedID = np.array(predsRed.id)
        
        predsRed = np.array(predsRed.cancer)
        
        return predsRed, predsRedID
 
#%% Import methods

class tfMod(tfHelpers):
    """
    Imported PP functions required for loading /node extraction. 
    Any changed methods overloaded here
    
    """
    # In this version load mask, then get data from mask or non-mask data 
    # Give paths, so hardcode choices for now
    def loadPPFilesV7(paths, files, cParams, labels, nodeMode=1):
        """
        Load PPed files with either PPV6 or UNET based nodes
        Replaces loadPPFilesV6, which is not included here.
        """
        if isinstance(files, str):
            files = [files]
        elif isinstance(files, pd.DataFrame):
            files = set(files.iloc[:,0])
        else:
            files = set(files)
    
        nFiles = len(files)
        
        xBuff = cParams['xBuff'] # in each direction (size=*2)
        yBuff = cParams['yBuff'] # in each direction
        zBuff = cParams['zBuff'] # in each direction
        nMax = cParams['nMax']
        
        loaded = np.empty(shape=[0,zBuff*2,yBuff*2,xBuff*2,1])
        newLabels = pd.DataFrame(columns=['id', 'cancer'])  
        
        
        path2 = paths['PPedSSD']
    
        path1 = paths['PPedV6SSD']
        path3 = paths['UNETPredPPV1']
        
        r = 0     
        for fn in files:
            print('Attempting to load data: ' + path1+fn )    
            
            data2 = np.load(path2+fn)['resImage3D']
    
            
            if nodeMode == 1:
                # Use simple nodes
                print('Attempting to load nodes: ' + path1+fn)
                data1 = np.load(path1+fn)['arr_0']
                
            elif nodeMode == 2:
                print('Attempting to load nodes: ' + path3+fn)
                # Use UNET nodes
                data1 = np.load(path3+fn)['nodPreds'].astype(np.float32)
                data1 = np.squeeze(data1)
                print(np.sum(data1))
    
    
            # c, cLabs = getNodesAndLabels([data], \
            #                    cParams=cParams, labels=labels.iloc[r,:])
            
            # Get candidaes
            smData, n, xyzMin, xyzMax = extractNodes2(
                                data1=data1, data2=data2, buffs=[xBuff,yBuff,zBuff], 
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

    def loadTraining():
        """
        Script to load training data and split in to validation sets.
        TODO: 
            Move params, make general with loadTest()
        """
        cParams = {'xBuff' : 20, # in each direction (size=*2)
                   'yBuff' : 20, # in each direction
                   'zBuff' : 10, # in each direction
                   'nMax' : 20}
           
        # image_size = 100
        num_channels = 1
        num_labels = 2
        validProp = 0.15
        # Image size
        xSize = cParams['xBuff']*2 # Input size
        ySize = cParams['yBuff']*2 # Input size
        zSize = cParams['zBuff']*2 # Input size, depth
             
        allFiles, trainFiles, testFiles, _, nTrainFiles, _ = \
                availFiles(paths['PPedSSD'])
        
        nLoad = nTrainFiles
        #nLoad = 100
        loadIdx = np.random.choice(range(0,trainFiles.shape[0]), nLoad, replace=False)
        splitLoad = round(nLoad*(0-validProp))
        trainIdx = loadIdx[0:splitLoad]
        validIdx = loadIdx[splitLoad:nLoad]
        
        trainFilesLoad = trainFiles.iloc[trainIdx]                                   
        trainLabelsLoad = labelsCSV.iloc[trainIdx]
        cTrain, cTrainLabels = loadPPFilesV7(paths,
                              trainFilesLoad, cParams, trainLabelsLoad, nodeMode=1)
        
        validFilesLoad = trainFiles.iloc[validIdx]                                   
        validLabelsLoad = labelsCSV.iloc[validIdx]
        cValid, cValidLabels = loadPPFilesV7(paths,
                              validFilesLoad, cParams, validLabelsLoad, nodeMode=1)
                              

    def loadTest():
    # Load labels to workspace alongside filenames
        labelsCSV = pd.read_csv(paths['labels_s12'])
        submission = pd.read_csv(paths['SampleSub_s2'])
        
        #%% Load test data
        
        allFiles, trainFiles, testFiles, _, nTrainFiles, _ = \
                availFiles(paths['PPed_s12'])
        
        # Should match training
        cParams = {'xBuff' : 20, # in each direction (size=*2)
                   'yBuff' : 20, # in each direction
                   'zBuff' : 10, # in each direction
                   'nMax' : 10}
        
        # image_size = 100
        num_channels = 1
        num_labels = 2
        validProp = 0.15
        # Image size
        xSize = cParams['xBuff']*2 # Input size
        ySize = cParams['yBuff']*2 # Input size
        zSize = cParams['zBuff']*2 # Input size, depth
                   
                   
        testFilesLoad = testFiles
        #testLabels = np.zeros(shape=[testFiles.shape[0],1], dtype=np.int16)                                   
        # testLabels are all zeros placeholder                                
        cTest, cTestLabels = loadPPFilesV7(paths,
                              testFilesLoad, cParams, submission, nodeMode=2)
        
        # Recount 
        nTest = cTestLabels.shape[0]

        
    def variable_summaries(var, tag):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'+tag):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
    def convLayer(inputTensor, chanIn=1, chanOut=80, patchSize=5, 
                  strides=[1, 2, 2, 2, 1], layerName='conv', act='relu'):
      # Apply name scope
      with tf.name_scope(layerName):
          # Variables.
          # Weights
          w = tf.Variable(tf.truncated_normal(
              [patchSize, patchSize, patchSize, chanIn, chanOut], stddev=0.1), 
                name='W')
          tfMod.variable_summaries(w, layerName)
          # Biases
          b = tf.Variable(tf.zeros([chanOut]), name='B')
          
          # Op
          conv = tf.nn.conv3d(inputTensor, w, strides, padding='SAME')
          
          if act == 'relu':
              act = tf.nn.relu(conv + b)
          elif act == 'tanh':
              act = tf.nn.tanh(conv + b)
          elif act == 'drop':
              act = tf.nn.dropout(conv + b, 0.8)
              
          return act
      
              
    def fcLayer(inputTensor, chanIn, chanOut, layerName='fc', act='drop'):
      
        with tf.name_scope(layerName):
          w = tf.Variable(tf.truncated_normal([chanIn, chanOut], stddev=0.1), 
                          name='W')
          tfMod.variable_summaries(w, layerName)
          b = tf.Variable(tf.constant(1.0, shape=[chanOut]), name='B')
          
          if act == 'drop':
              act = tf.nn.dropout(tf.matmul(inputTensor, w) + b, 0.8)
          elif act == 'relu':
              act = tf.nn.relu(tf.matmul(inputTensor, w) + b)
            
          return act
              
    def buildNetwork(mode='train'):
        """ 
         Function to build the training or test network
         2 conv layers, 1 fully connected layer
         Only differences is placeholder names/sizes
         Train: trainSubset (batch size), validSubset (valid size)
         Test: testSubset (50)
         
         TODO: 
             Generalise to work with test or train
             (differences is placeholder size)
             Move params.
             Particularly outChans, which is varied between models
         """
        
        if mode == 'train':
            batch_size = 360# this is n rows expected by placeholder - files produce multiple rows
            
        elif mode == 'test':
            batch_size = 50
            
        
        
        nBatch = batch_size
        patch_size = 5
        # depth = 1
        num_hidden = 560
        nTest = 1216#testFiles.shape[0]
        nTrainBatch = round(nBatch*(1-validProp))
        nValidBatch = round(nBatch*validProp)
        nValidBatch = 160
        # nValid = cValid.shape[0]*0.5
        nValid = 200
        graph = tf.Graph()
        
        
        with graph.as_default():

          # Input data.
          trainSubset = tf.placeholder(tf.float32, shape=(nTrainBatch, 
                                        zSize, ySize, xSize, num_channels), 
                                      name='PHTrain')
                  
          trainSubsetLabels = tf.placeholder(tf.float32, shape=(nTrainBatch,2), 
                                             name='PHTrainLabs')
          
          validSubset = tf.placeholder(tf.float32, shape=(nValidBatch, 
                               zSize, ySize, xSize, num_channels), 
                              name='PHValid')
          
          validSubsetLabels = tf.placeholder(tf.float32, shape=(nValidBatch,2), 
                                             name='PHValidLabs')
          
          testDataset = tf.placeholder(tf.float32, shape = (nTest, 
                                    zSize, ySize, xSize, num_channels), 
                                              name='PHTest')
          
          LR = tf.Variable(1)
          outChans = 80
          
          # Layer sizes not defined in functions
          layer4_weights = tf.Variable(tf.truncated_normal(
              [num_hidden, num_labels], stddev=0.1))
        
          layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
          
          # Model.
          def model(smData, outChans=80, patchSize=5,
                    fcNum = num_hidden, mName=''):
             
            act1 = convLayer(smData, chanIn=1, chanOut=outChans,
                             strides=[1, 2, 2, 2, 1], 
                            patchSize=patchSize, 
                             layerName=mName+'conv1', act='tanh')
            
            act2 = convLayer(act1, chanIn=outChans, chanOut=outChans, 
                             strides=[1, 2, 2, 2, 1],
                             patchSize=patchSize, 
                             layerName=mName+'conv2', act='drop')
            
            # Reshape for FC
            shape = act2.get_shape().as_list()
            linSize = shape[1] * shape[2] * shape[3] * shape[4]
            reshape = tf.reshape(act2, [shape[0], linSize])
        
            act3 = fcLayer(reshape, chanIn=linSize, chanOut=fcNum, 
                          layerName=mName+'fc1', act='drop')
        
            output = tf.matmul(act3, layer4_weights) + layer4_biases
            
            return output
          
          # Training computation.
          with tf.name_scope("Models"):  
              logits = model(trainSubset, outChans=outChans, patchSize=patch_size, 
                         fcNum=num_hidden, mName='train') 
              logitsV = model(validSubset, outChans=outChans, patchSize=patch_size, 
                         fcNum=num_hidden, mName='valid')
          
          with tf.name_scope("xent"):      
              
              loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=trainSubsetLabels, 
                                                    logits=logits))
              
              lossV = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=validSubsetLabels, 
                                                        logits=logitsV))
              
          # Optimizer.
          with tf.name_scope("train"):
              optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
          
          # Predictions for the training, validation, and test data.
          with tf.name_scope("Preds"):
              train_prediction = tf.nn.softmax(logits)
              valid_prediction = tf.nn.softmax(logitsV)
          
              test_prediction = tf.nn.softmax(model(testDataset, outChans=outChans, 
                                                    patchSize=patch_size, 
                                                    fcNum=num_hidden))
          
          with tf.name_scope("Accuracy"):
              
              cp = tf.equal(tf.argmax(logits,1), tf.argmax(trainSubsetLabels,1))
              trAcc2 = tf.reduce_mean(tf.cast(cp, tf.float32))
              
              cp2 = tf.equal(tf.argmax(logitsV,1), tf.argmax(validSubsetLabels,1))
              vAcc2 = tf.reduce_mean(tf.cast(cp2, tf.float32))
              
          # Add ops to save and restore all the variables.
          saver = tf.train.Saver()
          # Add summary writer
         
          tf.summary.scalar("va", vAcc2)
          # Create a summary to monitor accuracy tensor
          tf.summary.scalar("ta", trAcc2)
          variable_summaries(trAcc2, 'TrainingAcc')
          variable_summaries(vAcc2, 'ValidAcc')
          variable_summaries(lossV, 'Trainingloss')
          variable_summaries(loss, 'Validloss')
          merged_summary = tf.summary.merge_all()
          
          
    def runTraining():
        """
        Function to run the training network.
        Output for Tensorboard.
        Chekpoints/model to disk
        TODO:
            Move params/paths.
            Tidy
        """
        num_steps = 240000
        initialRate = 0.000001
        initialRate = 0.001
        
        scalingFactor = 6
        
        rand.seed(123123)
        
        tsAcc = []
        vsAcc = []
        tLoss = []
        vLoss = []
        it = -1
        bestPerf = 0
        sinceLast = 0
        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            
            tf.global_variables_initializer().run()
            
            logDir = "Summary/4"
            #writer = tf.summary.FileWriter(logDir)
            writer = tf.summary.FileWriter(logDir, graph=tf.get_default_graph())
            #writer.add_graph(session.graph)
            writer.flush()
            
            print('Initialized')
        
            for step in range(num_steps):
                it+=1       
               
                learningRate = initialRate/np.exp(it/(num_steps/scalingFactor))        
                
                # Get data and labels
                # allData -> trainLoad -> this batch
                fIdxTrain = np.random.choice(range(0, cTrain.shape[0]), 
                                                   nTrainBatch, 
                                          replace=False)
                fIdxValid = np.random.choice(range(0, cValid.shape[0]), 
                                                   nValidBatch,
                                          replace=False)
        
                
                # New: Data set already loaded
                cTrainSubset = cTrain[fIdxTrain,:,:,:,:]
                cTrainLabelsSubset = cTrainLabels['cancer'].iloc[fIdxTrain]
                cValidSubset = cValid[fIdxValid,:,:,:,:]
                cValidLabelsSubset = cValidLabels['cancer'].iloc[fIdxValid]
                
                feed_dict = {trainSubset : cTrainSubset, 
                             trainSubsetLabels : oneHot(cTrainLabelsSubset,2),
                             validSubset : cValidSubset, 
                             validSubsetLabels : oneHot(cValidLabelsSubset,2)}
                             # LR : learningRate}
                             # testDataset : testData}
                             
                _, l, predictions, lv, lp, ta, va, s = session.run(
                          [optimizer, loss, train_prediction, 
                           lossV, valid_prediction, trAcc2, vAcc2, merged_summary], 
                          feed_dict=feed_dict)
                
                if not step % 20:
                    print('Adding summary??')
        
                    # s = session.run(merged_summary)
                    writer.add_run_metadata(tf.RunMetadata(), 'step%d' % step)
                    writer.add_summary(s,step)
                    #writer.add_summary(variable_summaries(ta),step)
                    #writer.add_summary(variable_summaries(va),step)
                    #writer.add_summary(variable_summaries(lv),step)
                    #writer.add_summary(variable_summaries(lp),step)
                    writer.flush()
                    
                # if (step % 50 == 0):
                print('-------------------------------------------------------------')
                print('LR: ' + str(learningRate))
                print('Minibatch loss at step %d: %f' % (step, l))
                
                tsAcc.append(accuracy(predictions, oneHot(cTrainLabelsSubset)))
                tLoss.append(l)
                print("Training accuracy: " + str(tsAcc[it]))
                
                # lv, lp = valid_prediction.eval(feed_dict=feed_dict)
                
                vsAcc.append(accuracy(lp,oneHot(cValidLabelsSubset)))
                vLoss.append(lv)
                
                print("Validation accuracy: " + str(vsAcc[it]))
                print("Validation loss: " + str(lv))
                
                
                print('Training acc:')
                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                
                plt.plot(tsAcc)
                plt.plot(vsAcc, c='r')
                # plt.title('A tale of 2 subplots')
                plt.ylabel('Accuracy')
                plt.xlabel('Step')
        
                print('Training loss:')
                plt.subplot(1,2,2)
                l1 = plt.plot(tLoss, label='Train')
                l2 = plt.plot(vLoss, c='r', label='Valid')
                plt.legend(labels=['Train', 'Valid'])
                plt.ylabel('Loss')
                plt.xlabel('Step')
                
                plt.show()
        
               
                # Save the variables to disk.
                # Rolling average of 5 steps
                sinceLast +=1
                if step > 5:
                    # Current perf is this step and previous 4
                    currentPerf = np.mean(vsAcc[it-5:it])
                    # If this is better than the bestPerf, update bestPerf and 
                    # checkpoint model
                    if currentPerf >= bestPerf and sinceLast>2:
                        sinceLast = 0
                        bestPerf=currentPerf
                        chkFn = logDir + "model_st" + str(step) + '_' + \
                        str(round(float(currentPerf),3)) + ".ckpt"
        
                        # save_path = saver.save(session, chkFn)
                        save_path = saver.save(session, logDir+"/model.ckpt", step)
                        
                        print("Model saved in file: %s" % save_path)          
        
            # Save the variables to disk.
            #save_path = saver.save(session, paths['TF3D_40'])
            #print("Model saved in file: %s" % save_path)   
             
            
        print('Training acc:')
        plt.plot(tsAcc)
        plt.show()
        print('Validation acc:')
        plt.plot(vsAcc)
        plt.show()
        
    def runPrediction(data):
        """
        Run prediction
        Graph should be test graph from buildNetwwork
        Runs on batches of 50
        
        TODO:
            Generalise for test and train predictions
        """
        
        # Create graph. Remember to set outChans.
        # HERE
        
        #n = cTrain.shape[0]
        n = cTest.shape[0]
        
        with tf.Session(graph=graph) as session:
          # Restore variables from disk.
          saver.restore(session, paths['TF3D_2'])
          #saver.restore(session, paths['TF3D_TB4'])
          #saver.restore(session, "Models/Summary4")
          # saver.restore(session, paths['TF3D_1'])
          # Remeber to change outChans
          print("Model restored.")
          
          preds = np.empty(shape = (n,2))
          sIdx = 0
          eIdx = 0
          for ni in range(0,int(n/50)+1):
              
              sIdx = eIdx
              eIdx = sIdx+50
              
              # If overshooting, TF still exects 50 so get 50 even if some already done
              if eIdx > n:
                  eIdx = n
                  sIdx = eIdx-50
              
              print(str(sIdx) + ' to ' + str(eIdx))
              
              #feed_dict = {testDataset : cTrain[sIdx:eIdx,:,:,:,:].astype(np.float32)}
              feed_dict = {testDataset : cTest[sIdx:eIdx,:,:,:,:].astype(np.float32)}
              preds[sIdx:eIdx] = test_prediction.eval(feed_dict=feed_dict)
              
              print("Prediction done.")    

    def saveSub(preds, samSubPath, fn):
        """
        Load sample submission
        Reduce predictions
        Save submission
        """
        submission = pd.read_csv(samSubPath)
        
        redPreds, redPredsID = tfHelpers.reducePreds(preds, cTestLabels)
        submission['cancer'] = redPreds
        submission.to_csv(fn, index=False)  


    def savePreds(preds, fn):
        """ 
        Reduce predictions
        Save predictions alongside ID
        """
        
        redPreds, redPredsID = tfHelpers.reducePreds(preds, cTrainLabels)
        
        trainSub = pd.DataFrame()
        trainSub['id'] = redPredsID
        trainSub['cancer'] = redPreds
        trainSub.to_csv(fn, index=False) 
        
        
