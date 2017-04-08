# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:15:38 2017

@author: Gareth
"""

#%% Imports

import os


#%% 

class paths():
    def __init__(self):
        self.paths = { # Data
                 'Raw': "R:\\NoneDrive\\DSB2017 Data\\stage1\\",
                 'Raw_s2': "R:\\NoneDrive\\DSB2017 Data\\stage2\\",
                 'PPed': "S:\\OneDrive\\MATLAB\\DSB2017\\Stage1PP\\",
                 'PPed_s2': "S:\\OneDrive\\MATLAB\\DSB2017\\Stage2PP\\",
                 'PPedSSD': "C:\\Stage1PP\\",
                 'PPedSSD_s2': "C:\\Stage2PP\\",
                 'PPedV4' : 'C:\\Stage1PPV4\\',
                 'PPedV4_s2' : 'C:\\Stage2PPV4\\',
                 'PPedV4SSD' : 'C:\\Stage1PPV4SSD\\',
                 'PPedV4SSD_s2' : 'C:\\Stage2PPV4SSD\\',
                 'PPedV6' : "R:\\NoneDrive\\DSB2017 Data\\Stage1PPedV6\\",
                 'PPedV6_s2' : "R:\\NoneDrive\\DSB2017 Data\\Stage2PPedV6\\",
                 'labels': "stage1_labels.csv",
                 'labels_s2': "stage2_labels.csv",
                 'SampleSub': "stage1_sample_submission.csv",
                 'SampleSub_s2': "stage2_sample_submission.csv",
                 # LUNA
                 'UNETLoad': 'R:\\NoneDrive\\DSB2017 Data\\LUNA16\\All\\',
                 'UNETSave': 'S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\',
                 'UNETCands': 'S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\CSVFILES\\candidates.csv',
                 'UNETAno': 'S:\\OneDrive\\Matlab\\DSB2017\\LUNA16\\CSVFILES\\annotations.csv',
                  # Models
                 'TF3D_40' : "S:\\OneDrive\\Matlab\DSB2017\\Kaggle-DSB2017\\tmp\\Finalmodel_TF3D_PPV1_40.ckpt",
                 'TF3D_40_s2' : "",
                 'UNETTrained' : 'S:\\OneDrive\\Matlab\\DSB2017\\Kaggle-DSB2017\\Models\\KerasUNET256256Seq19.hdf5',
                 # Predictions
                 'UNETPredPPV1' : 'R:\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV1\\',
                 'UNETPredPPV1_s2' : '',
                 'UNETPredPPV6' : 'R:\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPV6\\',
                 'UNETPredPPV6_s2' : "",
                 'UNETPredPPS2V2' : 'R:\\NoneDrive\\DSB2017 Data\\UNETPred\\UNETPPS2V2\\',
                 'UNETPredPPS2V2_s2' : ""
                 }
                 
        """ 
        Run through paths and check files are availbale and if directories 
        exist. If files don't exist, do nothing. If directories don't exist,
        create. Some paths may be empty - skip these (reports "file not 
        available".)
        """
        for p, v in self.paths.items():
            if len(v)>2 and v[-1] == '\\':
                if os.path.isdir(v):
                   print('Dir exists: ' + v + '   (' + p + ')')
                else:
                    print('Creating dir: ' + v + '   (' + p + ')')
                    os.mkdir(v)
            else:
                if os.path.isfile(v):
                    print('File exists: ' + v + '   (' + p + ')')
                else:
                    print('File not available: ' + v + '   (' + p + ')')
                               