# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:15:38 2017

Stage 1 included training and test data
IDed using stage1_labels.csv - if not in labels, part of training set


Stage 2 includes test data only
Also inlcudes labels for set 1. Have combined this in to stage2_labels.csv. 
Should mean sets can be mereged, and IDed as previously. Ie., not in stage2 
labels, part of test set. 

Have created combined sets of PPV1 and PPV6 in Stage12 folders, use 
stage12_labels.csv

@author: Gareth
"""

#%% Imports

import os
import builtins as __builtin__


#%% 



class paths():
    """
    Create specified defualt paths depending on computer.
    """
    
    def __init__(self, ndd="R"):
       # ndd is drive letter for storage drive 
        self.paths = { # Data
                 "Raw": ndd+":\\NoneDrive\\DSB2017 Data\\stage1\\",
                 "Raw_s2": ndd+":\\NoneDrive\\DSB2017 Data\\stage2\\",
                 "PPed": "S:\\OneDrive\\MATLAB\\DSB2017\\Stage1PP\\",
                 "PPed_s2": "S:\\OneDrive\\MATLAB\\DSB2017\\Stage2PP\\",
                 "PPed_s12": "S:\\OneDrive\\MATLAB\\DSB2017\\Stage12PP\\",
                 "PPedSSD": "C:\\Stage1PP\\",
                 "PPedSSD_s2": "C:\\Stage2PP\\",
                 "PPedV4" : "C:\\Stage1PPV4\\",
                 "PPedV4_s2" : "C:\\Stage2PPV4\\",
                 "PPedV4SSD" : "C:\\Stage1PPV4SSD\\",
                 "PPedV4SSD_s2" : "C:\\Stage2PPV4SSD\\",
                 "PPedV6" : ndd+":\\NoneDrive\\DSB2017 Data\\Stage1PPedV6\\",
                 "PPedV6_s2" : ndd+":\\NoneDrive\\DSB2017 Data\\Stage2PPedV6\\",
                 "PPedV6_s12" : ndd+":\\NoneDrive\\DSB2017 Data\\Stage12PPedV6\\",
                 "labels": "stage1_labels.csv",
                 "labels_s12": "stage12_labels.csv",
                 "SampleSub": "stage1_sample_submission.csv",
                 "SampleSub_s2": "stage2_sample_submission.csv",
                 # LUNA
                 "UNETLoad": ndd+":\\NoneDrive\\DSB2017 Data\\LUNA16\\All\\",
                 "UNETSave": "S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\",
                 "UNETCands": "S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\CSVFILES\\candidates.csv",
                 "UNETAno": "S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\CSVFILES\\annotations.csv",
                  # Models
                 "TF3D_40" : "S:\\OneDrive\\Matlab\DSB2017\\Kaggle-DSB2017\\tmp\\Finalmodel_TF3D_PPV1_40.ckpt",
                 "TF3D_40_s2" : "",
                 "UNETTrained" : "S:\\OneDrive\\Matlab\\DSB2017\\Kaggle-DSB-2017\\Models\\KerasUNET256256Seq19.hdf5",
                 "UNETFromTutorial" : "S:\\OneDrive\\Matlab\\DSB2017\\DSB3Tutorial\\tutorial_code\\unetTut.hdf5",
                 # Predictions
                 "UNETPredPPV1" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1\\",
                 "UNETPredPPV1_s2" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1_s2\\",
                 "UNETPredPPV1_s12" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1_s12\\",
                 "UNETPredPPV1_UT" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1_UT\\",
                 "UNETPredPPV1_UT_s2" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1_UT_s2\\",
                 "UNETPredPPV6" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV6\\",
                 "UNETPredPPV6_s2" : "",
                 "UNETPredPPS2V2" : ndd+":\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPS2V2\\",
                 "UNETPredPPS2V2_s2" : ""
                 }
                 
        paths.disp(self, write=1)
        

    def disp(self, write=0):
        """ 
        Run through paths and check files are availbale and if directories 
        exist. If files don"t exist, do nothing. If directories don"t exist,
        create if write==1. Some paths may be empty - skip these 
        (reports "file not  
        available".)
        """
        for p, v in self.paths.items():
            if len(v)>2 and v[-1] == "\\":
                if os.path.isdir(v):
                   __builtin__.print("Dir exists: " + v + "   (" + p + ")")
                else:
                    if write:
                        os.mkdir(v)
                        __builtin__.print("Dir created: " + v + "   (" + p + ")")
                    else:
                        __builtin__.print("Dir missing: " + v + "   (" + p + ")")
            else:
                if os.path.isfile(v):
                    __builtin__.print("File exists: " + v + "   (" + p + ")")
                else:
                    __builtin__.print("File not available: " + v + "   (" + p + ")")     
                    