# -*- coding: utf-8 -*-
"""

PPV1
References
Preprocessing and plotting:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/discussion



PPV6
Preprocessing: Segmentation, candidate nodule detection using LUNA16 dataset and UNet model
Based on https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing

Currentely added:
1) Segementation - batch processing and saving to PPedV6
2) Column based candidate search prototypes - no batch here, do in minbatch get
smData, n, xyzMin, xyzMax = extractNodes(data). 




"""

#%% Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc



#%% Helper functions
"""
Helpers are functions used by different PP methods. Some are modified, some are
redundent.
"""

class helpers:
    @staticmethod
    def read_ct_scan(folder_name): # Used by PPV6
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))        
        
        try:
            st = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        except:
            st = slices[0].SliceLocation - slices[1].SliceLocation

        if st < 0:
            # Order appears to be reversed, resrot the otehr way round
            print('Flipping')
            slices.sort(key = lambda x: int(0-x.InstanceNumber))        
            
        for s in slices:
            s.SliceThickness = abs(st)    
        
        
        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices
    
    @staticmethod
    def read_ct_scan_HU(folder_name): # Used by PPV6
        # Read the slices from the dicom file
        slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))        
        
        try:
            st = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        except:
            st = slices[0].SliceLocation - slices[1].SliceLocation

        if st < 0:
            # Order appears to be reversed, resrot the otehr way round
            print('Flipping')
            slices.sort(key = lambda x: int(0-x.InstanceNumber))        
            
        for s in slices:
            s.SliceThickness = abs(st)    
        
        
        image3D = ppV6.toHU(slices)
        
        return image3D
    
    @staticmethod    
    def plot_ct_scan(scan): # Used by PPV6
        f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
        for i in range(0, scan.shape[0], 5):
            plots[int(i / 20), int((i % 20) / 5)].axis('off')
            plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
            
    @staticmethod
    def get_segmented_lungs(im, lm=604, plot=False):
        
        '''
        This funtion segments the lungs from the given 2D slice.
        '''
        if plot == True:
            f, plots = plt.subplots(8, 1, figsize=(5, 40))
        '''
        Step 1: Convert into a binary image. 
        '''
        binary = im < lm
        if plot == True:
            plots[0].axis('off')
            plots[0].imshow(binary, cmap=plt.cm.bone) 
        '''
        Step 2: Remove the blobs connected to the border of the image.
        '''
        cleared = clear_border(binary)
        if plot == True:
            plots[1].axis('off')
            plots[1].imshow(cleared, cmap=plt.cm.bone) 
        '''
        Step 3: Label the image.
        '''
        label_image = label(cleared)
        if plot == True:
            plots[2].axis('off')
            plots[2].imshow(label_image, cmap=plt.cm.bone) 
        '''
        Step 4: Keep the labels with 2 largest areas.
        '''
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                           label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0
        if plot == True:
            plots[3].axis('off')
            plots[3].imshow(binary, cmap=plt.cm.bone) 
        '''
        Step 5: Erosion operation with a disk of radius 2. This operation is 
        seperate the lung nodules attached to the blood vessels.
        '''
        selem = disk(2)
        binary = binary_erosion(binary, selem)
        if plot == True:
            plots[4].axis('off')
            plots[4].imshow(binary, cmap=plt.cm.bone) 
        '''
        Step 6: Closure operation with a disk of radius 10. This operation is 
        to keep nodules attached to the lung wall.
        '''
        selem = disk(10)
        binary = binary_closing(binary, selem)
        if plot == True:
            plots[5].axis('off')
            plots[5].imshow(binary, cmap=plt.cm.bone) 
        '''
        Step 7: Fill in the small holes inside the binary mask of lungs.
        '''
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)
        if plot == True:
            plots[6].axis('off')
            plots[6].imshow(binary, cmap=plt.cm.bone) 
        '''
        Step 8: Superimpose the binary mask on the input image.
        '''
        get_high_vals = binary == 0
        im[get_high_vals] = 0
        if plot == True:
            plots[7].axis('off')
            plots[7].imshow(im, cmap=plt.cm.bone) 
            
        return im
    
    @staticmethod   
    def segment_lung_from_ct_scan(ct_scan, lm=604): # Used by PPV6
        
        return np.asarray([ppV6.get_segmented_lungs(slice, lm=lm) for slice in ct_scan])    
    
    @staticmethod    
    def plot_3d(image, threshold=-300): # Used by PPV6
        
        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)
        p = p[:,:,::-1]
        
        verts, faces = measure.marching_cubes(p, threshold)
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])
    
        plt.show()    
   
    @staticmethod
    def filterBVs(segmented_ct_scan): # Used by PPV6, PPV1
        selem = ball(2)
        binary = binary_closing(segmented_ct_scan, selem)
        
        label_scan = label(binary)
        
        areas = [r.area for r in regionprops(label_scan)]
        areas.sort()
        
        for r in regionprops(label_scan):
            max_x, max_y, max_z = 0, 0, 0
            min_x, min_y, min_z = 1000, 1000, 1000
            
            for c in r.coords:
                max_z = max(c[0], max_z)
                max_y = max(c[1], max_y)
                max_x = max(c[2], max_x)
                
                min_z = min(c[0], min_z)
                min_y = min(c[1], min_y)
                min_x = min(c[2], min_x)
                
                # Below was areas[-3]
            if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[max(-len(areas), -3)]):
                for c in r.coords:
                    segmented_ct_scan[c[0], c[1], c[2]] = 0
            else:
                pass
               # index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
        return segmented_ct_scan
    
    # Take list of patients slices, convert to Hounsfield units
    @staticmethod    
    def toHU2_PPV6(slices, limit=-2000): # Used by PPV6 (not used?)
        image3D = np.stack([s.pixel_array for s in slices])
        
        # Pixels outside of scan are -2000. Set these to air (0)
        image3D[image3D<=limit] = 0
        
        # Meta data contains 0 setting for this scanner
        inter = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        
        if slope != 1:
            image3D = slope * image3D.astype(np.float64)
            image3D = image3D.astype(np.int16)
        
        image3D = np.array(image3D, dtype=np.int16)
        image3D += np.int16(inter)  
        # return image3D as 3D np array
        return(image3D)
        
    # Take list of patients slices, convert to Hounsfield units
    @staticmethod
    def toHU(slices, limit=-2000): # Used by PPV1
        image3D = np.stack([s.pixel_array for s in slices])
        
        # Pixels outside of scan are -2000. Set these to air (0)
        image3D[image3D<=limit] = 0
        
        # Meta data contains 0 setting for this scanner
        inter = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        
        if slope != 1:
            image3D = slope * image3D.astype(np.float64)
            image3D = image3D.astype(np.int16)
        
        image3D = np.array(image3D, dtype=np.int16)
        image3D += np.int16(inter)  
        # return image3D as 3D np array
        return(image3D.astype(np.int16))

    @staticmethod
    def load(path): # Used by PPV1
        # Load all images from specified folder
        # Find slices    
        sliceNames = os.listdir(path)
        # Load in to list
        slices = []
        for fn in sliceNames:
            
            try:
                slices.append(dicom.read_file(path+fn))
            except: 
                # Eg 912 last file
                print(path+fn + ' appears invalid')
                
                   
        if len(sliceNames)<2:
            print('Not running - not enough data')
            return(slices, False)
        else:
            # Sort loaded slices
            slices.sort(key = lambda x: int(x.InstanceNumber))
            
            # Calculate slice thickness
            #try:
            #    st = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)    
            #except:
            #    st = 1
            #    print("Warning - no .SliceLocation!")
                
            try:
                st = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
            except:
                st = slices[0].SliceLocation - slices[1].SliceLocation
                
            if st < 0:
                # Order appears to be reversed, resrot the otehr way round
                print('Flipping')
                slices.sort(key = lambda x: int(0-x.InstanceNumber))        
                
            for s in slices:
                s.SliceThickness = abs(st)
                
            # Return list of slices, containing image and meta data
            # Image is in .pixel_array
            return(slices, True)
    
    @staticmethod
    def resample(slices, image3D, newSpacing = [1,1,1]): # Used by PPV1
        # Work out current spacing
    
        spacing = np.array([slices[0].SliceThickness, 
                            slices[0].PixelSpacing[0], 
                            slices[0].PixelSpacing[1]])
                            
        factor = spacing / newSpacing
        newShapeReal = image3D.shape*factor
        newShapeRound = np.round(newShapeReal)
        factorReal = newShapeRound / image3D.shape                    
        newSpacing = spacing / factorReal
        
        # Interpolate in 3D
        # Slow - 5 seconds
        image3D = scipy.ndimage.interpolation.zoom(image3D, factorReal)
        
        return(image3D.astype(np.int16), newSpacing)            
                
    @staticmethod
    def plot3D(image, thresh = -300): # Used by PPV1
        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)
        p = p[:,:,::-1]
        
        verts, faces = measure.marching_cubes(p, thresh)
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
    
        ax.set_xlim(0, p.shape[0])  # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(0, p.shape[1])  # b = 10
        ax.set_zlim(0, p.shape[2])  # c = 16
    
        plt.show()    
        
    @staticmethod
    def largest_label_volume(im, bg=-1): # Used by PPV1
        vals, counts = np.unique(im, return_counts=True)
    
        counts = counts[vals != bg]
        vals = vals[vals != bg]
    
        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None
    
    @staticmethod        
    def segment_lung_mask(image, fill_lung_structures=True): # Used by PPV1
    
        # not actually binary, but 1 and 2. 
        # 0 is treated as background, which we do not want
        binary_image = np.array(image > -320, dtype=np.int8)+1
        labels = measure.label(binary_image)
        
        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air 
        #   around the person in half
        background_label = labels[0,0,0]
        
        #Fill the air around the person
        binary_image[background_label == labels] = 2
        
        
        # Method of filling the lung structures (that is superior to something like 
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = helpers.largest_label_volume(labeling, bg=0)
                
                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1
    
        
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = helpers.largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
     
        return binary_image
        
#%% PPV1 class
"""
Does PP based on Guido's script
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/discussion
Uses [PPed] path

"""
    
class ppV1(helpers):
    def __init__(self, paRaw, paPP, params):
        self.paRaw = paRaw
        self.paPP = paPP
        self.params = params
 
    def process(self, patients): # ppPatients from preprocessing.py
        # Needs to handle difference between lists and [lists]
        if isinstance(patients, str):
            patients = [patients]
            
        n = len(patients)
        i = 0
        
        for pi in patients:
            do = True
            i+=1
            print("Processing patient: " + pi + ' (' + str(i) + '/' + str(n) + ')')
            if os.path.isfile(self.paPP+pi+".npz") and not self.params['forcePP']:
                print('   Already done.')
                # But only skip if ok
                if not ppV1.verify(self.paPP+pi+".npz"):
                    do = True
                else:
                    do = False
                    code = False
                    
            # Doesn't exist or corrupted        
            if do:
                # Load
                print('   Loading...')
                slices, code = ppV1.load(self.paRaw+pi+'\\')
            
            # And continue only if data actually exists
            # (Load returns flase if there isn't enough raw data)
            if code:
                # Convert to HU
                print('   Converting to HU...')
                image3D = ppV1.toHU(slices, -2000)
                # Resample image
                print('   Resampling...')
                resImage3D, newSpacing = ppV1.resample(slices, 
                                                  image3D, newSpacing=[1,1,1])
                                                  
                segmented_lungs = ppV1.segment_lung_mask(resImage3D, False)
                segmented_lungs_fill = ppV1.segment_lung_mask(resImage3D, True)
                
                diff = segmented_lungs_fill-segmented_lungs          
                diff2 = ppV1.filterBVs(diff)            
                
                # Save to disk
                print('   Saving...')
                np.savez_compressed(self.paPP+pi, 
                                    resImage3D=resImage3D, 
                                    segmented_lungs=segmented_lungs, 
                                    segmented_lungs_fill=segmented_lungs_fill, 
                                    diff=diff,
                                    diff2=diff2)
                
                if self.params['plotHist']:
                    print('   Plotting hist...')
                    plt.hist(image3D.flatten(), bins=80)
                    plt.show()
                if self.params['plotMid2D']:
                    print('   Plotting 2D...')
                    plt.imshow(resImage3D[round(resImage3D.shape[0]/2)])
                    plt.show()
                if self.params['plot3D']:
                    print('   Plotting 3D...')
                    ppV1.plot3D(resImage3D, 400)
                    plt.show()
                    ppV1.plot3D(segmented_lungs, 0)
                    plt.show()
                    ppV1.plot3D(segmented_lungs_fill, 0)
                    plt.show()
                    ppV1.plot3D(segmented_lungs_fill - segmented_lungs, 0)
                    plt.show()
                print('Done' + ' (' + str(np.round(i/n*100,2)) + '%)')
            
            print('-'*20)

    def verify(path):
        print('   Verifying: ' + path)
        try:    
            l = np.load(path)
            resImage3D = l['resImage3D']
            segmented_lungs = l['segmented_lungs']
            segmented_lungs_fill = l['segmented_lungs_fill']
            diff = l['diff']
            diff2 = l['diff2']
            
            print('   Passed.')
            return(True)
        except: 
            print('   Verification failed.')
            return(False)     

                                                                           
#%% PPV6 Class
"""
Does PP with segmentation - based on ArnavJan's script

Init method
Load method & Verify method
Batch process method
Uses helpers:
read_ct_scan
read_ct_scan_HU
plot_ct_scan
segment_lung_from_ct_scan
plot_3d
filterBVs
toHU2
"""

class ppV6(helpers):
    def __init__(self, paRaw, paPP, params):
        self.paRaw = paRaw
        self.paPP = paPP
        self.params = params
        
    def loadPPFilesV6(path, files): 
        # Possibly not correct version - from preprocessV6, 
        # likely updated in later scripts 
        if isinstance(files, str):
            files = [files]
        elif isinstance(files, pd.DataFrame):
            files = set(files.iloc[:,0])
        else:
            files = set(files)
            
        nFiles = len(files)
        loaded = np.zeros([nFiles, dims[0], dims[1], dims[2]], dtype = np.float16)        
            
        for fn in files:
            data = np.load(path+fn)['arr_0']

        return(data.astype(np.int16))

    def verify(path):
        print('   Verifying: ' + path)
        try:    
            data = np.load(path)['arr_0']
            del(data)
            print('   Passed.')
            return(True)
        except: 
            print('   Verification failed.')
            return(False)

        
    def process(self, patients): # (Replaces ppPatientsV6)
        
        if isinstance(patients, str):
            patients = [patients]
                    
        n = len(patients)
        i = 0
        
        for pi in patients:
            do = True
            i+=1
            print("Processing patient: " + pi + 
                    ' (' + str(i) + '/' + str(n) + ')')
            if os.path.isfile(self.paPP+pi+".npz") and not self.params['forcePP']:
                print('   Already done.')
                # But only skip if ok
                if not ppV6.verify(self.paPP+pi+".npz"):
                    do = True
                else:
                    do = False
                    
            # Doesn't exist or corrupted        
            if do:
                # Load
                print('   Loading...')
                image3D = ppV6.read_ct_scan(self.paRaw+pi+'\\')
            
                # Segment
                print('   Segmenting...')
                resImage3D = ppV6.segment_lung_from_ct_scan(image3D)
                
                # Filter blood vessels
                print('   Filtering...')
                resImage3D = ppV6.filterBVs(resImage3D)            
                
                # Save to disk
                print('   Saving...')
                np.savez_compressed(self.paPP+pi, resImage3D)
                
                # Output plots        
                if self.params['plotHist']:
                    print('   Plotting hist...')
                    plt.hist(resImage3D.flatten(), bins=80)
                    plt.show()
                if self.params['plotMid2D']:
                    print('   Plotting 2D...')
                    plt.imshow(resImage3D[round(resImage3D.shape[0]/2)])
                    plt.show()
                if self.params['plotCTScan']:
                    ppV6.plot_ct_scan(resImage3D)
                if self.params['plot3D']:
                    try:
                        print('   Plotting 3D...')
                        ppV6.plot_3d(resImage3D, 400)
                        plt.show()
                    except:
                        print('    Plot failed')
                print('Done' + ' (' + str(np.round(i/n*100,2)) + '%)')
        
        print('-'*20)
        
