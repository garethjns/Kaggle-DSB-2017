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


#%% Helpers
# A few functions that do the remaning PP steps to the data after loading
# from disk

class tfHelpers(fHelpers):
    def normalize(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image>1] = 1.
        image[image<0] = 0.
        return image.astype(np.float16)
        
    
    def zero_center(image, PIXEL_MEAN = 0.25):
        image = image - PIXEL_MEAN
        return image.astype(np.float16)

    
    # Overload fHelpers.getData with this version from trainV3.
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
            
            
#%% TF Mod
""" 
Class handelling the tensorFlow model

.buildTrainingNetwork builds network for training
.runTraining runs the taining
.builtTestNetwork (to do) builds same network for predicting, but different placeholders
.runTest() will be from predict script (not added yet)  
"""

class tfMod(tfHelpers):
    def __init__(self, modPath, dataPath, params):
        self.modPath = modPath
        self.dataPath = dataPath
        self.params = params
    
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])    
            
    def availFiles(path):
        # Find preprocess files
        availFiles = pd.DataFrame(os.listdir(path))
        nAvailFiles = availFiles.shape[0]
        # Find those that are in the training set
        labelsCSV = pd.read_csv(paths['labels'])
        
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
    def loadPPFilesV6(path, files, cParams, labels):
        
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
            smData, n, xyzMin, xyzMax = extractNodes(
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
    def loadPPFilesV7(paths, files, cParams, labels, nodeMode=1):
        
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
            smData, n, xyzMin, xyzMax = extractNodes(
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
        
    def buildTrainingNetwork():
        # Build network 
        # 2 conv layers, 1 fully connected layer
        batch_size = 50 # this is n rows expected by placeholder - files produce multiple rows
        nBatch = batch_size
        patch_size = 5
        # depth = 1
        num_hidden = 560
        nTest = 1216#testFiles.shape[0]
        nTrain = round(nBatch*(1-validProp))
        # nValid = cValid.shape[0]*0.5
        nValid = 200
        graph = tf.Graph()
        
        with graph.as_default():
        
          # Input data.
          trainSubset = tf.placeholder(tf.float32, shape=(nTrain, 
                               zSize, ySize, xSize, num_channels), name='PHTrain')
          trainSubsetLabels = tf.placeholder(tf.float32, shape=(nTrain,2), name='PHTrainLabs')
          
          validSubset = tf.placeholder(tf.float32, shape=(nValid, 
                               zSize, ySize, xSize, num_channels), name='PHValid')
          validSubsetLabels = tf.placeholder(tf.float32, shape=(nValid,2), name='PHValidLabs')
          
          testDataset = tf.placeholder(tf.float32, shape = (nTest, 
                                                 zSize, ySize, xSize, num_channels), name='PHTest')
          
          # testDataset = tf.Variable(initial_value = [1.0,1.0,1.0,1.0,1.0], validate_shape=False, name='PHTest')
          LR = tf.Variable(1)
          # testDataset = tf.constant(testData)
          
          outChans = 80
          
          # Variables.
          layer1_weights = tf.Variable(tf.truncated_normal(
              [patch_size, patch_size, patch_size, num_channels, outChans], stddev=0.1))
        
          layer1_biases = tf.Variable(tf.zeros([outChans]))
          
          layer2_weights = tf.Variable(tf.truncated_normal(
              [patch_size, patch_size, patch_size, outChans, outChans], stddev=0.1))
        
          layer2_biases = tf.Variable(tf.constant(1.0, shape=[outChans]))
          
          # layer2b_weights = tf.Variable(tf.truncated_normal(
          #    [patch_size, patch_size, patch_size, outChans, outChans], stddev=0.1))
        
          # layer2b_biases = tf.Variable(tf.constant(1.0, shape=[outChans]))
        
          layer3_weights = tf.Variable(tf.truncated_normal(
              [xSize // 4 * ySize // 4 * zSize //4 * outChans, num_hidden], stddev=0.1))
        
          layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        
          layer4_weights = tf.Variable(tf.truncated_normal(
              [num_hidden, num_labels], stddev=0.1))
        
          layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
          
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
          logits = model(trainSubset) 
          loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=trainSubsetLabels, logits=logits))
          
          logitsV = model(validSubset)
          lossV = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=validSubsetLabels, logits=logitsV))
          
          # Optimizer.
          optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss)
          
          # Predictions for the training, validation, and test data.
          train_prediction = tf.nn.softmax(logits)
          valid_prediction = tf.nn.softmax(logitsV)
          
          test_prediction = model(testDataset)
          test_prediction = tf.nn.softmax(model(testDataset))
          
          # Add ops to save and restore all the variables.
          saver = tf.train.Saver()
    def runTraining():
        #%% Run training - load all first
        num_steps = 60000
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
            print('Initialized')
            #print(session.run(logits))
            for step in range(num_steps):
                it+=1       
               
                learningRate = initialRate/np.exp(it/(num_steps/scalingFactor))        
                
                # Get data and labels
                # allData -> trainLoad -> this batch
                fIdxTrain = np.random.choice(range(0, cTrain.shape[0]), 
                                                   round(nBatch*(1-validProp)), 
                                          replace=False)
                fIdxValid = np.random.choice(range(0, cValid.shape[0]), 
                                                   round(nValid),
                                          replace=False)
        
                
                # New: Data set already loaded
                cTrainSubset = cTrain[fIdxTrain,:,:,:,:]
                cTrainLabelsSubset = cTrainLabels['cancer'].iloc[fIdxTrain]
                cValidSubset = cValid[fIdxValid,:,:,:,:]
                cValidLabelsSubset = cValidLabels['cancer'].iloc[fIdxValid]
                # cValidSubset = cValid
                # cValidLabelsSubset = cValidLabels['cancer']
        
                # This is inefficient at the moment
                # Loading and processing more than enough data each step
                # As unknown how many rows from each file
                # Then cropping to fit expected nBatch
                # Would be better to move this processing to before training
                #
                # Extract inds from list:
                # tss = [trainSubsetLoad[i] for i in fIdxTrain]
                # Process subset
                # trainSubsetData, trainLabelsSubset = \
                #             getNodesAndLabels(tss, \
                #                    cParams=cParams, labels=trainLabelsLoad.iloc[fIdxTrain])
                # Strip DF info from labels                
                # trainLabelsSubset = trainLabelsSubset['cancer']
                #
                # Extract inds from list:
                # vss = [validSubsetLoad[i] for i in fIdxValid]
                # Process subset
                # validSubsetData, validLabelsSubset = \
                #             getNodesAndLabels(vss, \
                #                   cParams=cParams, labels=validLabelsLoad.iloc[fIdxValid])
                # Strip DF info from labels                
                # validLabelsSubset = validLabelsSubset['cancer']
                #
                #
                # For now crop to size expected by placeholder
                # trainLabelsSubset = trainLabelsSubset[0:round(nBatch*(1-validProp))]
                # trainsubsetData = trainSubsetData[0:round(nBatch*(1-validProp)),:,:,:,:]
                # validLabelsSubset = validLabelsSubset[0:round(nBatch*validProp)]
                # validSubsetData = validSubsetData[0:round(nBatch*validProp),:,:,:,:]
              
                feed_dict = {trainSubset : cTrainSubset, 
                             trainSubsetLabels : oneHot(cTrainLabelsSubset,2),
                             validSubset : cValidSubset, 
                             validSubsetLabels : oneHot(cValidLabelsSubset,2),
                             LR : learningRate}
                             # testDataset : testData}
                             
                _, l, predictions, lv, lp = session.run(
                [optimizer, loss, train_prediction, lossV, valid_prediction], feed_dict=feed_dict)
                
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
                # plt.show()
                
                
                print('Training loss:')
                plt.subplot(1,2,2)
                l1 = plt.plot(tLoss, label='Train')
                l2 = plt.plot(vLoss, c='r', label='Valid')
                plt.legend(labels=['Train', 'Valid'])
                plt.ylabel('Loss')
                plt.xlabel('Step')
                
                plt.show()
                
                #print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                #print('Validation accuracy: %.1f%%' % accuracy(
                #     valid_prediction.eval(), valid_labels))
                #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
               
                # Save the variables to disk.
                # Rolling average of 5 steps
                sinceLast +=1
                if step > 5:
                    # Current perf is this step and previous 4
                    currentPerf = np.mean(vsAcc[it-5:it])
                    # If this is better than the bestPerf, update bestPerf and 
                    # checkpoint model
                    if currentPerf >= bestPerf and sinceLast>10:
                        sinceLast = 0
                        bestPerf=currentPerf
                        chkFn = "S:\OneDrive\Matlab\DSB2017\Kaggle-DSB2017\\tmp\\model_st" + str(step) + '_' + \
                        str(round(float(currentPerf),3)) + ".ckpt"
        
                        save_path = saver.save(session, chkFn)
                        
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
