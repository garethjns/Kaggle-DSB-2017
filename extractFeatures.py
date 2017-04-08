# -*- coding: utf-8 -*-
"""
Classes to generate features for XGB models
Features generates training and test tables
Contains versions from generateFeatures and generateFeaturesV3 

fHelpers contains the functions to do the work. If they were modified between 
GF and GF3, both versions included. V3 version renamed _V3.
"""

#%% Imports

from preprocessing import plot_3d, plot3D


#%% Helpers
class fHelpers():
    def availFiles(path): # Used by V1,V3
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
           
    
# This function is stupid
def getHist(data, bins=80, xLims=[-5000,5000]): # Used by V1, V3
    
    out = np.zeros(shape=[16], dtype=np.float16)
    
    # Get hist    
    y,x = np.histogram(data.flatten(),bins)
    
    # Limit
    x=x[1::]    
    
    y = y[x>xLims[0]]
    x = x[x>xLims[0]]

    y = y[x<xLims[1]]
    x = x[x<xLims[1]]
    
    if len(y)<16:
        mInd = len(y)
    elif len(y)>=16:
        mInd=16
  
    out[0:mInd] = y[0:mInd]

    return out,x
    
# Get the "column" coordinates
def getXY(data, nMax=999, plotOn=False): # Used by V1, V3
    # Reduce Z
    mData = np.mean(data, axis=0)
    
    image_max = ndi.maximum_filter(mData, size=20, mode='constant')
    coordsxy = np.array(peak_local_max(mData, min_distance=10))
    
    # Sort by mag
    mag = mData[coordsxy[:,0],coordsxy[:,1]]
    
    if plotOn:
        plt.plot(mag)
        
    sIdx = np.argsort(mag)
    # Descending
    sIdx = sIdx[::-1]
    # Limit by max requested
    nKeep = min(len(sIdx), nMax)
    sIdx = sIdx[0:nKeep]    
    
    # Sort coords
    coordsxy = coordsxy[sIdx,:]
    
    # For now, limit to three coords
    # coordsxy = coordsxy[0:3,0:2]
    
    # Plot coords
    if plotOn:
        plt.plot(mag[sIdx])
        plt.show()        
        
        plt.imshow(mData, cmap=plt.cm.gray)
        plt.scatter(coordsxy[:,1],coordsxy[:,0])
        plt.show()
    
    return coordsxy, mData
    

# Plot the columns to search using buffs = [xBuff, yBuff]    
def getCols(mData, coordsxy, buffs=[20,20], plotOn=False): # Used by V1, V3
    # Plot coords
    if plotOn:
        plt.imshow(mData, cmap=plt.cm.gray)
        plt.scatter(coordsxy[:,1],coordsxy[:,0])
        plt.show()
    
    # Create columns
    # For now, limit to three coords
    # coordsxy = coordsxy[0:3,0:2]
    
    xBuff = buffs[0] # in each direction
    yBuff = buffs[1] # in each direction
    # zBuff = 10 # in each direction - not used here
    
    # x appears to be in col 1, y in col 0
    xMin = coordsxy[:,1]-xBuff 
    yMin = coordsxy[:,0]-yBuff 
    xMax = coordsxy[:,1]+xBuff 
    yMax = coordsxy[:,0]+yBuff 
    #[xMin, xMax, yMin, yMax]
    
    if plotOn:
        plt.imshow(mData, cmap=plt.cm.gray)
        for c in range(0,len(xMin)):
            
            pltArr = np.zeros(shape=[5,2])
            pltArr[0,0] = xMin[c]
            pltArr[0,1] = yMin[c]
            pltArr[1,0] = xMax[c]
            pltArr[1,1] = yMin[c]
            pltArr[2,0] = xMax[c]
            pltArr[2,1] = yMax[c]
            pltArr[3,0] = xMin[c]
            pltArr[3,1] = yMax[c]  
            pltArr[4,0] = xMin[c]
            pltArr[4,1] = yMin[c]  
        
        
            plt.plot(pltArr[:,0], pltArr[:,1])
        plt.show()
    
    return xMin, xMax, yMin, yMax


    def getZ(data, xMin, xMax, yMin, yMax, zBuff, plotOn=True):
        # Save the xyz for each z peak found in:
        xyzMin = np.empty(shape=[0,3]) # n x x,y,z
        xyzMax = np.empty(shape=[0,3])
        lims = np.array(data.shape)[::-1]
        r= -1
        for c in range(0,len(xMin)):
            # Remeber, data is [z,y,x]
            col = data[:,yMin[c]:yMax[c],xMin[c]:xMax[c]]
            # Average x and y
            mCol = np.mean(col, axis=1)
            z = np.mean(mCol, axis=1)
            
            # Check params of this
            zIdx = signal.find_peaks_cwt(z, np.arange(1,6))    
            
            if plotOn:
                plt.plot(z)
                plt.scatter(zIdx, z[zIdx])
                
                
            # Now for each xIdx, buffer out 10 pixels
            zMin = np.empty(shape=0)
            zMax = np.empty(shape=0)
            for zi in range(0,len(zIdx)):
                r +=1
                zMin = np.append(zMin, zIdx[zi]-zBuff)
                zMax = np.append(zMax, zIdx[zi]+zBuff)
                # zMin[zi] = zIdx[zi]-10
                # zMax[zi] = zIdx[zi]+10
                
                addMin = np.array([xMin[c], yMin[c], zMin[zi]]) # x,y,z
                addMax = np.array([xMax[c], yMax[c], zMax[zi]]) # x,y,z
                # Lims is also x,y,z (already converted from z,y,x)
                
                # Only append to final list of within range        
                if all(addMin >= 0) and all(addMax <= lims):
                    xyzMin = np.vstack([xyzMin, addMin])
                    xyzMax = np.vstack([xyzMax, addMax])
        
        if plotOn:    
            plt.show()  
            
        xyzMin = xyzMin.astype(np.int16)
        xyzMax = xyzMax.astype(np.int16)
    
        return xyzMin, xyzMax
    
    # Replaces above, gets data. Now handles empty. Uses plot_3d.
    def getData(data, xyzMin, xyzMax, buffs=[20,20,10], plotOn=False): # Used by V1, V3
        
        # Note buff sized used here must be specified (no data to calculate)
        if len(xyzMin) == 0:
            print('No data available')
            # For now, if now candidates are available, return 1 cube of zeros 
            # for this patient
            out = np.zeros(shape=[1,buffs[2],buffs[1],buffs[0]], dtype=np.int16)
          
            return out, 0
        
        # Note buff size is recalcualted here. Specified is ignored.
        n = len(xyzMin)
        xSize = int(xyzMax[0,0] - xyzMin[0,0])
        ySize = int(xyzMax[0,1] - xyzMin[0,1])
        zSize = int(xyzMax[0,2] - xyzMin[0,2])
        
        out = np.zeros(shape=[n,zSize,ySize,xSize], dtype=np.int16)
        for ni in range(0,n):
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
    
    
    def getGhost(pData): # Used by V1, V3
    
        # Get average planes    
        # Also load mask predictions from UNET
        mAx0 = np.mean(pData, axis=0)
        mAx1 = np.mean(pData, axis=1)  
        mAx2 = np.mean(pData, axis=2)      
        
        nInAx = pData.shape
    
        # Add XY plane to Z
        for s in range(0,nInAx[0]):
            pData[s,:,:] = pData[s,:,:] + mAx0
        # Add XZ plane to Y    
        for s in range(0,nInAx[1]):
            pData[:,s,:] = pData[:,s,:] + mAx1
        # Add YZ plane to X
        for s in range(0,nInAx[2]):
            pData[:,:,s] = pData[:,:,s] + mAx2
        
        # Finish average
        pData = np.round(pData/4)
                
        return pData
    
        
    def placeData(data, rel=False, dims=[100,100,100]): # Used by V1, V3
            """ For each axis
            Either:
            If same size, place straight in
            If file bigger, crop and fill output
            If output bigger, centre file in preallo -2000s
            
            If rel, calc dims as data dims + dims
            If Abs, use dims as specified
            """
            
            if rel:
                dims = np.array(data.shape, dtype=int)+dims
            
            dims=dims.astype(int)        
            
            loaded = np.zeros([dims[0], dims[1], dims[2]], dtype=np.int16)        
            
            axIn = list()
            axOut = list()
            # Lists[z,x,y], where each = list of start, end 
            for ax in range(0,3):
                axLen = data.shape[ax]
                if axLen == dims[ax]:
                    # Same size
                    axIn.append([0, int(dims[0])])
                    axOut.append([0, int(dims[0])]) 
    
                if  axLen < dims[ax]:
                    # File data smaller than output - buffer
                    axDiff = round((dims[ax] - axLen) / 2)
                  
                    axIn.append([0, int(axLen)])
                    axOut.append([int(axDiff), int(axDiff+axLen)])
               
                if axLen > dims[ax]:
                    # File data larger than output - crop from middle 
                    axDiff = (dims[ax] - axLen) / 2
                    # Note - rounding is weird
                    axIn.append([int(abs(round(axDiff))), int(round(axLen+axDiff))])
                    axOut.append([0, int(dims[ax])])
           
            # Place in output
            loaded[\
            axOut[0][0]:axOut[0][1], \
            axOut[1][0]:axOut[1][1], \
            axOut[2][0]:axOut[2][1]] = \
            data[\
            axIn[0][0]:axIn[0][1], \
            axIn[1][0]:axIn[1][1], \
            axIn[2][0]:axIn[2][1]]
    
            return loaded
            
            
    def extractNodes(data, buffs=[20,20,10], nMax = 50, plotOn=False): # Used by V1, V3 (?)
    
        # Get coords
       
        coordsxy, mData = getXY(data, nMax=nMax, plotOn=plotOn) 
        
        # Get cols
        xMin, xMax, yMin, yMax = getCols(
                    mData, coordsxy, buffs=[buffs[0],buffs[1]], plotOn=plotOn)
                    
        # Get Z            
        xyzMin, xyzMax = getZ(data, xMin, xMax, yMin, yMax, buffs[2], plotOn=plotOn)
        
        # Get data (if possible) as [n, z, y, x]
        smData, n = getData(
                            data, xyzMin, xyzMax, 
                            buffs=[buffs[0],buffs[1],buffs[2]], plotOn=False)
                
                
        return smData, n, xyzMin, xyzMax
    
        
    # Input 2 versios of data - first to find, second to get    
    def extractNodes2(data1, data2, buffs=[1,1,1], nMax = 50, plotOn=False): # Used by V1, V3
    
        # Get coords
        coordsxy, mData = getXY(data1, nMax=nMax, plotOn=plotOn) 
        
        # Get cols
        xMin, xMax, yMin, yMax = getCols(
                mData, coordsxy, buffs=[buffs[0],buffs[1]], plotOn=plotOn)
                
        # Get Z            
        xyzMin, xyzMax = getZ(data1, xMin, xMax, yMin, yMax, buffs[2], plotOn=plotOn)
        
        # If there is a difference in size between data 1 and 2, translate data1s
        # coordinates from above in to the size of data2
        # For example, data1 from PPedV1 - was resampled using slice info
        # data2 from PPedV6 was left as nSx512x512, but the blood vessel removal
        # may have been better
        if data1.shape!=data2.shape:
            xyzMin2 = np.zeros(shape = xyzMin.shape, dtype = np.int16)
            xyzMax2 = np.zeros(shape = xyzMax.shape, dtype = np.int16)
            # Resize coords
            for c in range(0,3):
                # Data.shape is zyx, xyzMin is xyz
                xF = data2.shape[2-c]/data1.shape[2-c]
                xyzMin2[:,c] = xyzMin[:,c]*xF # already int16
                xyzMax2[:,c] = xyzMin2[:,c] + buffs[c]*2
        else:
            xyzMin2 = xyzMin
            xyzMax2 = xyzMax
        
        # Get data (if possible) as [n, z, y, x]
        smData, n = getData(
                        data2, xyzMin2, xyzMax2, 
                        buffs=[buffs[0],buffs[1],buffs[2]], plotOn=plotOn)
            
            
        return smData, n, xyzMin, xyzMax    
    
    # Uses extractNodes1 (ie 1 data input)    
    def addFeatures(data, mask=None, plotOn=False): # Used by V1, V3
           
        if not mask is None:
            # Apply mask
            data[mask.astype(np.bool)==False] = -2000
                 
            
        smData, cN, xyzMin, xyzMax = \
            extractNodes(data, buffs=[20,20,10], nMax = 100, plotOn=plotOn)
        
        # Prop top half
        xyzMid = (xyzMin[:,2]+xyzMax[:,2])/2
        mid = len(data[:,1,1])/2
        cPropTop = np.sum(xyzMid>mid)/len(xyzMid)
        
        # mean, median, std, x,y,z locs
        cMean = np.mean(xyzMid)
        cMedian = np.median(xyzMid)
        cStd = np.std(xyzMid)
        
        return cN, cPropTop, cMean, cMedian, cStd 
     
    
    # Uses extractNodes2 (ie 2 data input and size of cords corrected if different
    # - as in trainV3)    
    def addFeatures2(data1, data2, mask=None, plotOn=False): # Used by V1, V3
           
        if not mask is None:
            # Apply mask
            data2[mask.astype(np.bool)==False] = -2000
                 
        # Input 2 versios of data - first to find, second to get     
        smData, cN, xyzMin, xyzMax = \
            extractNodes2(data1.astype(np.int64), data2.astype(np.int64), buffs=[20,20,10], nMax = 100, plotOn=plotOn)
        
        # Prop top half
        xyzMid = (xyzMin[:,2]+xyzMax[:,2])/2
        mid = len(data1[:,1,1])/2
        cPropTop = np.sum(xyzMid>mid)/len(xyzMid)
        
        # mean, median, std, x,y,z locs
        cMean = np.mean(xyzMid)
        cMedian = np.median(xyzMid)
        cStd = np.std(xyzMid)
        
        return cN, cPropTop, cMean, cMedian, cStd     
        
    def appendRow(row=pd.DataFrame(), data=[], names=[], ind=0, disp=False): # Used by V1, V3
       
        for d,n in zip(data,names):
            if disp:
                print(n)
                print(d)
            
            row = pd.concat([row, pd.DataFrame({n:d}, index=[0])], axis=1)
    
        return row           
    
#%% features
"""
Init - sets paths
loadAllPP - loads PP files from PPV1 and PPV6 and UNET predictions
loadAllPP_V3 
getRows - two versions, handles fHelpers
processBoth - processes both versions of getRows (might add)
save - save train and test tables

"""    
class features(fHelpers):
    def __init__(self, paPPV1, paPPV6, params):
        self.PPV1 = paPPV1
        self.PPV6 = paPPV6
        self.params = params
        self.trainTable
        self.testTable

    
    def loadAllPP(paths, fn, plotOn=False):    
        # Load all PPed files available for subject
    
        if isinstance(fn, pd.Series):
            fn = set(fn)
            fn = fn.pop()
    
    
        print('Attempting to load: ' +  str(fn))    
        
        dataV1 = np.load(paths['PPed']+str(fn))
        
        # dataV6 = dataV1['resImage3D']
        dataV6 = np.load(paths['PPedV6']+fn)['arr_0']
        
        UNM = np.load(paths['UNETPredPPV1']+str(fn))
        UNM = UNM['nodPreds']
       
        if plotOn:
            print('   Plotting hist...')
            plt.hist(dataV1['resImage3D'].flatten(), bins=80)
            plt.show()
            plt.hist(dataV6.flatten(), bins=80)
            plt.show()
    
            print('   Plotting 2D...')
            plt.imshow(dataV1['resImage3D'][round(dataV1['resImage3D'].shape[0]/2)])
            plt.show()
            plt.imshow(dataV6[round(dataV6.shape[0]/2)])
            plt.show()
                    
        return dataV1, dataV6, UNM
        
        def getRows(paths, files, labelsCSV, testFlag=False, plotOn=False):
    
    table = pd.DataFrame()
    nF = len(files)
    
    for r in range(0,nF):
    
        print('Appending row: ' + str(r+1) + '/' + str(nF))
        fn = files.iloc[r,0]
        
        # stupid part
        if testFlag:
            # id in placeholder labels has .npz
           nl = labelsCSV.loc[labelsCSV['id']==fn]
        else:
            nl = labelsCSV.loc[labelsCSV['id']==fn[0:-4]]
            
        label = nl.iloc[0][1]

        pData1, pData6, UNM = loadAllPP(paths, fn) 
        
        # Prepare dataFrame
        row = appendRow(data=[fn, label], 
                        names=['name', 'cancer'])
        
        #pData1 contains
        # resImage3D = pData1['resImage3D']
        # segmented_lungs = pData1['segmented_lungs']
        # segmented_lungs_fill = pData1['segmented_lungs_fill']
        # diff = pData1['diff']
        # diff2 = pData1['diff2']
        
        
        # Get ghost
        ghost = getGhost(pData1['resImage3D'])
        
        if plotOn:
            plt.imshow(ghost[100,:,:], cmap=plt.cm.gray)
            plt.show()
            plt.imshow(pData1['resImage3D'][100,:,:], cmap=plt.cm.gray)
            plt.show()
            gDiff = pData1['resImage3D']-ghost
            plt.imshow(gDiff[100,:,:], cmap=plt.cm.gray)
            plt.show()

        # Run get functions
        
        # Find candidates V1, mask: pData['segmented_lungs_fill']
        # Prop top half
        # mean, median, std, x,y,z locs
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['PD1_SLF_cN', 'PD1_SLF_cPT', 'PD1_SLF_cM', 
                        'PD1_SLF_cMed', 'PD1_SLF_cStd'], ind=r)
        
        # Find candidates V1, mask: pData['diff']
        data = 'resImage3D'
        mask = 'diff'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['PD1_Diff_cN', 'PD1_Diff_cPT', 'PD1_Diff_cM', 
                        'PD1_Diff_cMed', 'PD1_Diff_cStd'], ind=r)
        
        # Find candidates V1, mask: pData['diff2']
        data = 'resImage3D'
        mask = 'diff2'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['PD1_Diff2_cN', 'PD1_Diff2_cPT', 'PD1_Diff2_cM', 
                        'PD1_Diff2_cMed', 'PD1_Diff2_cStd'], ind=r)
        
        
        # Find candidates ghost, mask: pData['segmented_lungs']
        data = 'resImage3D'
        mask = 'segmented_lungs'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['PD1_SL_cN', 'PD1_SL_cPT', 'PD1_SL_cM', 
                        'PD1_SL_cMed', 'PD1_SL_cStd'], ind=r)
        
        # Find candidates ghost, mask: pData['diff']
        data = ghost
        mask = 'diff'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(ghost, pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['gh_Diff_cN', 'gh_Diff_cPT', 'gh_Diff_cM', 
                        'gh_Diff_cMed', 'gh_Diff_cStd'], ind=r)
        
        # Find candidates ghost, mask: pData['diff2']
        data = ghost
        mask = 'diff2'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(ghost, pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['gh_Diff2_cN', 'gh_Diff2_cPT', 'gh_Diff2_cM', 
                        'gh_Diff2_cMed', 'gh_Diff2_cStd'], ind=r)
        
        # Get lung volume from segmented_lung_fill?
        mask = 'segmented_lungs_fill'
        data = pData1[mask]
        lv = np.sum(data)
        row = appendRow(row=row, data=[lv], 
                        names=['SLF_LV'], ind=r)
        # Get height/volume ratio
        thresh = 400
        data1D = np.sum(data,axis=(1,2))
        
        data1D[data1D<thresh]=0
        data1D[data1D>=thresh]=1
        vhr = lv/np.sum(data1D)
        
        row = appendRow(row=row, data=[vhr], 
                        names=['VHR'], ind=r)        
        
        # Get histogram around 0 V1, mask: segmented_lungs
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        y,x = getHist(data[mask.astype(np.bool)], bins=50, xLims=[-400,400])
        # plt.plot(x,y)        

            
        
        row = appendRow(row=row, data=y, 
                        names=['hist1', 'hist2', 'hist3', 'hist4', 'hist5', 'hist6', 'hist7', 'hist8', 'hist9', 'hist10', 'hist11', 'hist12', 'hist13', 'hist14', 'hist15', 'hist16'], ind=r)
        
        
        # Get histogram around 0 ghost, mask: segmented_lungs
        # Estimate bone density (mean/median vals in bone range)
        data = 'resImage3D'
        data = pData1[data]
        m = np.mean(data[(data>=700) & (data<=3000)])
        st = np.std(data[(data>=700) & (data<=3000)])
        row = appendRow(row=row, data=[m, st], 
                        names=['boneMean', 'boanSTD'], ind=r)
        
        # Estimate blood volume and blood to lung vol ratio
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        
        data[(data<30)] = 0
        data[(data>45)] = 0
        data[(data>=30) & (data<=45)] = 1
        bSum = np.sum(data[mask.astype(np.bool)])
        bVR = bSum/lv
      
        row = appendRow(row=row, data=[bSum, bVR], 
                        names=['bloodVol', 'bloodVR'], ind=r)
                
        # Estimate air to volume ratio
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        
        data[(data>-1000)] = 0
        data[(data<-1999)] = 0
        data[(data>=-1999) & (data<=-1000) & mask.astype(np.bool)] = 1
        aSum = np.sum(data[mask.astype(np.bool)])
        aVR = aSum/lv
      
        row = appendRow(row=row, data=[aSum, aVR], 
                        names=['airVol', 'airVR'],  ind=r)
                
        
        # Estimate air to blood ratio
        abR = aSum/bSum
        abVR = aVR/bVR
        row = appendRow(row=row, data=[abR, abVR], 
                        names=['airBlood_R', 'airVRBloodVR_R'], ind=r)
        
        
        # Get ratios of these ranges to volume:
        # http://www.auntminnie.com/index.aspx?sec=ser&sub=def&pag=dis&ItemID=80701
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data1 = pData1[data]
        data2 = pData1[data]        
        mask = pData1[mask]
        # Less malignant range
        data1[data1<-196]=0
        data1[data1>42]=0
        lessMalSum = np.sum(data1[mask.astype(np.bool)])
        lessMalBinSum = np.sum(data1[mask.astype(np.bool)]==1)
        lessMalMed = np.median(data1[mask.astype(np.bool)])
        lessMalVR = lessMalSum/lv
        data2[data1<-94]=0
        data2[data1>176]=0
        moreMalSum = np.sum(data2[mask.astype(np.bool)])
        moreMalBinSum = np.sum(data2[mask.astype(np.bool)]==1)
        moreMalMed = np.median(data2[mask.astype(np.bool)])
        moreMalVR = moreMalSum/lv
        row = appendRow(row=row, 
                        data=[lessMalMed, lessMalBinSum, lessMalMed, lessMalVR, 
                                       moreMalSum, moreMalBinSum, moreMalMed, moreMalVR], 
                        names=['lessMalMedSLF', 'lessMalBinSumSLF', 'lessMalMedSLF', 'lessMalVRSLF', 
                               'moreMalSumSLF', 'moreMalBinSumSLF', 'moreMalMedSLF', 'moreMalVRSLF'], 
                        ind=r)
        
        # Also do some nodule finding on these two masks
        mask = 'segmented_lungs_fill'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(data1, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['lm_SLF_cN', 'lm_SLF_cPT', 'lm_SLF_cM', 
                        'lm_SLF_cMed', 'lm_SLF_cStd'], ind=r)
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(data2, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['mm_SLF_cN', 'mm_SLF_cPT', 'mm_SLF_cM', 
                        'mm_SLF_cMed', 'mm_SLF_cStd'], ind=r)
        
        # Use segmented lungs instead
        data = 'resImage3D'
        mask = 'segmented_lungs'
        data1 = pData1[data]
        data2 = pData1[data]        
        mask = pData1[mask]
        # Less malignant range
        data1[data1<-196]=0
        data1[data1>42]=0
        lessMalSum = np.sum(data1[mask.astype(np.bool)])
        lessMalBinSum = np.sum(data1[mask.astype(np.bool)]==1)
        lessMalMed = np.median(data1[mask.astype(np.bool)])
        lessMalVR = lessMalSum/lv
        data2[data1<-94]=0
        data2[data1>176]=0
        moreMalSum = np.sum(data2[mask.astype(np.bool)])
        moreMalBinSum = np.sum(data2[mask.astype(np.bool)]==1)
        moreMalMed = np.median(data2[mask.astype(np.bool)])
        moreMalVR = moreMalSum/lv
        row = appendRow(row=row, 
                        data=[lessMalMed, lessMalMed, lessMalVR, 
                                       moreMalSum, moreMalMed, moreMalVR], 
                        names=['lessMalMedSL', 'lessMalBinSumSL', 'lessMalMedSL', 'lessMalVRSL', 
                               'moreMalSumSL', 'moreMalBinSumSL', 'moreMalMedSL', 'moreMalVRSL'], 
                        ind=r)
        
        # Also do some nodule finding on these two masks
        mask = 'segmented_lungs'
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(data1, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['lm_SL_cN', 'lm_SL_cPT', 'lm_SL_cM', 
                        'lm_SL_cMed', 'lm_SL_cStd'], ind=r)
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures(data2, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                         names=['mm_SL_cN', 'mm_SL_cPT', 'mm_SL_cM', 
                        'mm_SL_cMed', 'mm_SL_cStd'], ind=r)
        
        
        # Get difference between mean and median V1, mask: filled
        # Get difference between mean and median ghost, mask: filled
        
        # From UNET
        data = UNM
        row = appendRow(row=row, data=[np.sum(UNM), np.max(UNM)], 
                        names=['UNETNodSum', 'UNETNodMax'], ind=r)
        
        
        
        # Assuming nodPreds are n x 1 x 256 x 256
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        # Nod mask
        nodMask = UNM.squeeze() # Remove extra dimension
        # Add features 2 uses extract features 2 (taken from trainV3) which uses
        # nodMask to find nods, then data and lung mask to extract
        cN, cPropTop, cMean, cMedian, cStd = \
        addFeatures2(nodMask, pData1[data], mask=pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd], 
                        names=['UNET_Diff_cN', 'UNET_Diff_cPT', 'UNET_Diff_cM', 
                        'UNET_Diff_cMed', 'UNET_Diff_cStd'], ind=r)
        # For later
        # Extract candidates for 3D network
        # Extract candidate similarity to other shapes
    
        print('Col check: ' + str(len(row.columns)) + ' for ' + fn)
        
        # Append row to table
        try:
            
            table = pd.concat([table, row], axis=0)
            print(row.columns)
        except:
            print('ERROR')
            print(row.columns)
            return row, table
        
    return table
    
    def getRows_V3(paths, files, labelsCSV, testFlag=False, plotOn=False):
    
    table = pd.DataFrame()
    nF = len(files)
    
    for r in range(0,nF):
    
        print('Appending row: ' + str(r+1) + '/' + str(nF))
        fn = files.iloc[r,0]
        
        # stupid part
        if testFlag:
            # id in placeholder labels has .npz
            nl = labelsCSV.loc[labelsCSV['id']==fn]
        else:
            nl = labelsCSV.loc[labelsCSV['id']==fn[0:-4]]
            
        label = nl.iloc[0][1]

        pData1, pData6, UNM = loadAllPP(paths, fn) 
        
        # Prepare dataFrame
        row = appendRow(data=[fn, label], 
                        names=['name', 'cancer'])
        
        #pData1 contains
        # resImage3D = pData1['resImage3D']
        # segmented_lungs = pData1['segmented_lungs']
        # segmented_lungs_fill = pData1['segmented_lungs_fill']
        # diff = pData1['diff']
        # diff2 = pData1['diff2']
        
        
        # Get ghost
        ghost = getGhost(pData1['resImage3D'])
        
        if plotOn:
            plt.imshow(ghost[100,:,:], cmap=plt.cm.gray)
            plt.show()
            plt.imshow(pData1['resImage3D'][100,:,:], cmap=plt.cm.gray)
            plt.show()
            gDiff = pData1['resImage3D']-ghost
            plt.imshow(gDiff[100,:,:], cmap=plt.cm.gray)
            plt.show()

        # Run get functions
        
        # Find candidates V1, mask: pData['segmented_lungs_fill']
        # Prop top half
        # mean, median, std, x,y,z locs
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['PD1_SLF_cN', 'PD1_SLF_cPT', 'PD1_SLF_cM', 
                        'PD1_SLF_cMed', 'PD1_SLF_cStd', 'PD1_SLF_cMT'], ind=r)
        
        # Find candidates V1, mask: pData['diff']
        data = 'resImage3D'
        mask = 'diff'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['PD1_Diff_cN', 'PD1_Diff_cPT', 'PD1_Diff_cM', 
                        'PD1_Diff_cMed', 'PD1_Diff_cStd', 'PD1_Diff_cMT'], ind=r)
        
        # Find candidates V1, mask: pData['diff2']
        data = 'resImage3D'
        mask = 'diff2'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['PD1_Diff2_cN', 'PD1_Diff2_cPT', 'PD1_Diff2_cM', 
                        'PD1_Diff2_cMed', 'PD1_Diff2_cStd', 'PD1_Diff2_cMT'], ind=r)
        
        
        # Find candidates ghost, mask: pData['segmented_lungs']
        data = 'resImage3D'
        mask = 'segmented_lungs'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(pData1[data], pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['PD1_SL_cN', 'PD1_SL_cPT', 'PD1_SL_cM', 
                        'PD1_SL_cMed', 'PD1_SL_cStd', 'PD1_SL_cMT'], ind=r)
        
        # Find candidates ghost, mask: pData['diff']
        data = ghost
        mask = 'diff'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(ghost, pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['gh_Diff_cN', 'gh_Diff_cPT', 'gh_Diff_cM', 
                        'gh_Diff_cMed', 'gh_Diff_cStd', 'gh_Diff_cMT'], ind=r)
        
        # Find candidates ghost, mask: pData['diff2']
        data = ghost
        mask = 'diff2'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(ghost, pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['gh_Diff2_cN', 'gh_Diff2_cPT', 'gh_Diff2_cM', 
                        'gh_Diff2_cMed', 'gh_Diff2_cStd', 'gh_Diff2_cMT'], ind=r)
        
        # Get lung volume from segmented_lung_fill?
        mask = 'segmented_lungs_fill'
        data = pData1[mask]
        lv = np.sum(data)
        row = appendRow(row=row, data=[lv], 
                        names=['SLF_LV'], ind=r)
        # Get height/volume ratio
        thresh = 400
        data1D = np.sum(data,axis=(1,2))
        
        data1D[data1D<thresh]=0
        data1D[data1D>=thresh]=1
        vhr = lv/np.sum(data1D)
        
        row = appendRow(row=row, data=[vhr], 
                        names=['VHR'], ind=r)        
        
        # Get histogram around 0 V1, mask: segmented_lungs
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        y,x = getHist(data[mask.astype(np.bool)], bins=50, xLims=[-400,400])
        # plt.plot(x,y)        

            
        row = appendRow(row=row, data=y, 
                        names=['hist1', 'hist2', 'hist3', 'hist4', 'hist5', 'hist6', 'hist7', 'hist8', 'hist9', 'hist10', 'hist11', 'hist12', 'hist13', 'hist14', 'hist15', 'hist16'], ind=r)
        
        
        # Get histogram around 0 ghost, mask: segmented_lungs
        # Estimate bone density (mean/median vals in bone range)
        data = 'resImage3D'
        data = pData1[data]
        m = np.mean(data[(data>=700) & (data<=3000)])
        st = np.std(data[(data>=700) & (data<=3000)])
        row = appendRow(row=row, data=[m, st], 
                        names=['boneMean', 'boanSTD'], ind=r)
        
        # Estimate blood volume and blood to lung vol ratio
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        
        data[(data<30)] = 0
        data[(data>45)] = 0
        data[(data>=30) & (data<=45)] = 1
        bSum = np.sum(data[mask.astype(np.bool)])
        bVR = bSum/lv
      
        row = appendRow(row=row, data=[bSum, bVR], 
                        names=['bloodVol', 'bloodVR'], ind=r)
                
        # Estimate air to volume ratio
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data = pData1[data]
        mask = pData1[mask]
        
        data[(data>-1000)] = 0
        data[(data<-1999)] = 0
        data[(data>=-1999) & (data<=-1000) & mask.astype(np.bool)] = 1
        aSum = np.sum(data[mask.astype(np.bool)])
        aVR = aSum/lv
      
        row = appendRow(row=row, data=[aSum, aVR], 
                        names=['airVol', 'airVR'],  ind=r)
                
        
        # Estimate air to blood ratio
        abR = aSum/bSum
        abVR = aVR/bVR
        row = appendRow(row=row, data=[abR, abVR], 
                        names=['airBlood_R', 'airVRBloodVR_R'], ind=r)
        
        
        # Get ratios of these ranges to volume:
        # http://www.auntminnie.com/index.aspx?sec=ser&sub=def&pag=dis&ItemID=80701
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        data1 = pData1[data]
        data2 = pData1[data]        
        mask = pData1[mask]
        # Less malignant range
        data1[data1<-196]=0
        data1[data1>42]=0
        lessMalSum = np.sum(data1[mask.astype(np.bool)])
        lessMalBinSum = np.sum(data1[mask.astype(np.bool)]==1)
        lessMalMed = np.median(data1[mask.astype(np.bool)])
        lessMalVR = lessMalSum/lv
        data2[data1<-94]=0
        data2[data1>176]=0
        moreMalSum = np.sum(data2[mask.astype(np.bool)])
        moreMalBinSum = np.sum(data2[mask.astype(np.bool)]==1)
        moreMalMed = np.median(data2[mask.astype(np.bool)])
        moreMalVR = moreMalSum/lv
        row = appendRow(row=row, 
                        data=[lessMalMed, lessMalBinSum, lessMalMed, lessMalVR, 
                                       moreMalSum, moreMalBinSum, moreMalMed, moreMalVR], 
                        names=['lessMalMedSLF', 'lessMalBinSumSLF', 'lessMalMedSLF', 'lessMalVRSLF', 
                               'moreMalSumSLF', 'moreMalBinSumSLF', 'moreMalMedSLF', 'moreMalVRSLF'], 
                        ind=r)
        
        # Also do some nodule finding on these two masks
        mask = 'segmented_lungs_fill'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(data1, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['lm_SLF_cN', 'lm_SLF_cPT', 'lm_SLF_cM', 
                        'lm_SLF_cMed', 'lm_SLF_cStd','lm_SLF_cMT'], ind=r)
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(data2, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['mm_SLF_cN', 'mm_SLF_cPT', 'mm_SLF_cM', 
                        'mm_SLF_cMed', 'mm_SLF_cStd', 'mm_SLF_cMT'], ind=r)
        
        # Use segmented lungs instead
        data = 'resImage3D'
        mask = 'segmented_lungs'
        data1 = pData1[data]
        data2 = pData1[data]        
        mask = pData1[mask]
        # Less malignant range
        data1[data1<-196]=0
        data1[data1>42]=0
        lessMalSum = np.sum(data1[mask.astype(np.bool)])
        lessMalBinSum = np.sum(data1[mask.astype(np.bool)]==1)
        lessMalMed = np.median(data1[mask.astype(np.bool)])
        lessMalVR = lessMalSum/lv
        data2[data1<-94]=0
        data2[data1>176]=0
        moreMalSum = np.sum(data2[mask.astype(np.bool)])
        moreMalBinSum = np.sum(data2[mask.astype(np.bool)]==1)
        moreMalMed = np.median(data2[mask.astype(np.bool)])
        moreMalVR = moreMalSum/lv
        row = appendRow(row=row, 
                        data=[lessMalMed, lessMalMed, lessMalVR, 
                                       moreMalSum, moreMalMed, moreMalVR], 
                        names=['lessMalMedSL', 'lessMalBinSumSL', 'lessMalMedSL', 'lessMalVRSL', 
                               'moreMalSumSL', 'moreMalBinSumSL', 'moreMalMedSL', 'moreMalVRSL'], 
                        ind=r)
                        
        
        # Also do some nodule finding on these two masks
        mask = 'segmented_lungs'
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(data1, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['lm_SL_cN', 'lm_SL_cPT', 'lm_SL_cM', 
                        'lm_SL_cMed', 'lm_SL_cStd', 'lm_SL_cMT'], ind=r)
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures(data2, pData1[mask], plotOn=plotOn) # data1 has new mask already applied
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                         names=['mm_SL_cN', 'mm_SL_cPT', 'mm_SL_cM', 
                        'mm_SL_cMed', 'mm_SL_cStd', 'mm_SL_cMT'], ind=r)
        
        
        # https://www.kaggle.com/c/data-science-bowl-2017/discussion/30686
        # Liver - size, prop of total vol, fat prop, mean, std as measure of inhomogeneity
        # (range: 40 to 60),
        # Fat (range: -150 to -50)
        # Est BMI
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        liver = pData1[data]
        fat = pData1[data]
        mask = pData1[mask]
        # Get rough liver. Limit to bottom two fiths.
        top = int(data1.shape[0]/5*2)
        liver[0:top,:,:]=0
        liver[data1<40]=0
        liver[data1>60]=0
        liverSum = np.sum(liver[mask.astype(np.bool)])
        liverMean = np.mean(liver[mask.astype(np.bool)])     
        liverStd = np.std(liver[mask.astype(np.bool)])  
        liverVol = np.sum(liver>0)
        liverLungProp = liverVol/lv       
        fat[data1<-150]=0
        fat[data1>-50]=0
        fatSum = np.sum(fat[mask.astype(np.bool)])
        fatMean = np.mean(fat[mask.astype(np.bool)])  
        fatStd = np.std(fat[mask.astype(np.bool)]) 
        fatVol = np.sum(fat>0)       
        fatLungProp = fatVol/lv    
        BMI = fatVol/data1.shape[0] # "BMI"
        row = appendRow(row=row, data=[liverSum, liverMean, liverStd, 
                                       liverVol, liverLungProp, fatSum,
                                       fatMean, fatStd, fatVol, fatLungProp, 
                                       BMI], 
                         names=['SLF_liver_sum', 'SLF_liver_mean',
                                'SLF_liver_std', 'SLF_liver_vol',
                                'SLF_liver_LP', 'SLF_fat_sum',
                                'SLF_fat_mean', 'SLF_fat_std', 'SLF_fat_vol',
                                'SLF_fat_LP', 'SLF_fat_BMI'], ind=r)
        # And same without mask
        data = 'resImage3D'
        liver = pData1[data]
        fat = pData1[data]
        mask = np.ones(data1.shape)
        # Get rough liver. Limit to bottom two fiths.
        top = int(data1.shape[0]/5*2)
        liver[0:top,:,:]=0
        liver[data1<40]=0
        liver[data1>60]=0
        liverSum = np.sum(liver[mask.astype(np.bool)])
        liverMean = np.mean(liver[mask.astype(np.bool)])     
        liverStd = np.std(liver[mask.astype(np.bool)])  
        liverVol = np.sum(liver>0)
        liverLungProp = liverVol/lv       
        fat[data1<-150]=0
        fat[data1>-50]=0
        fatSum = np.sum(fat[mask.astype(np.bool)])
        fatMean = np.mean(fat[mask.astype(np.bool)])  
        fatStd = np.std(fat[mask.astype(np.bool)]) 
        fatVol = np.sum(fat>0)       
        fatLungProp = fatVol/lv    
        BMI = fatVol/data1.shape[0] # "BMI"
        row = appendRow(row=row, data=[liverSum, liverMean, liverStd, 
                                       liverVol, liverLungProp, fatSum,
                                       fatMean, fatStd, fatVol, fatLungProp, 
                                       BMI], 
                         names=['nm_liver_sum', 'nm_liver_mean',
                                'nm_liver_std', 'nm_liver_vol',
                                'nm_liver_LP', 'nm_fat_sum',
                                'nm_fat_mean', 'nm_fat_std', 'nm_fat_vol',
                                'nm_fat_LP', 'nm_fat_BMI'], ind=r)
        
        # Get difference between mean and median V1, mask: filled
        # Get difference between mean and median ghost, mask: filled
        
        # From UNET
        data = UNM
        row = appendRow(row=row, data=[np.sum(UNM), np.max(UNM)], 
                        names=['UNETNodSum', 'UNETNodMax'], ind=r)
        
        
        
        # Assuming nodPreds are n x 1 x 256 x 256
        data = 'resImage3D'
        mask = 'segmented_lungs_fill'
        # Nod mask
        nodMask = UNM.squeeze() # Remove extra dimension
        # Add features 2 uses extract features 2 (taken from trainV3) which uses
        # nodMask to find nods, then data and lung mask to extract
        cN, cPropTop, cMean, cMedian, cStd, cMeanTop = \
        addFeatures2(nodMask, pData1[data], mask=pData1[mask], plotOn=plotOn)
        row = appendRow(row=row, data=[cN, cPropTop, cMean, cMedian, cStd, cMeanTop], 
                        names=['UNET_Diff_cN', 'UNET_Diff_cPT', 'UNET_Diff_cM', 
                        'UNET_Diff_cMed', 'UNET_Diff_cStd', 'UNET_Diff_cMT'], ind=r)
        # For later
        # Extract candidates for 3D network
        # Extract candidate similarity to other shapes
    
        print('Col check: ' + str(len(row.columns)) + ' for ' + fn)
        
        # Append row to table
        try:
            
            table = pd.concat([table, row], axis=0)
            print(row.columns)
        except:
            print('ERROR')
            print(row.columns)
            return row, table
        
    return table
    
