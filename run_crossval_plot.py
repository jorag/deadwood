#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:26:28 2019

@author: jorag
"""


import matplotlib.pyplot as plt
import numpy as np
import tkinter
from tkinter import filedialog
import os # Necessary for relative paths
import pickle # To load object
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# Name of input object and file with satellite data path string
dataset_use = 'vanZyl-B'
obj_in_name = dataset_use + '.pkl'
sat_pathfile_name = dataset_use + '-path'

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

knn_file = 'cross_validation_knn.pkl'
rf_file = 'cross_validation_rf.pkl'
                          
# Read result dicts - kNN
# Read predefined file
with open(os.path.join(dirname, 'data', knn_file )) as infile:
    knn_cv_all, knn_cv_sar, knn_cv_opt = pickle.load(infile)
    
# Read or create reult dicts - Random Forest
# Read predefined file
with open(os.path.join(dirname, 'data', rf_file )) as infile:
    rf_cv_all, rf_cv_sar, rf_cv_opt = pickle.load(infile)
    

# GET CROSS-VALIDATE RESULTS

# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])


# Cross validate - kNN - All data
knn_scores_all = knn_cv_all[dataset_use]
print('kNN OPT+SAR - ' + dataset_use + ' :')
print(knn_scores_all) 

# Cross validate - kNN - SAR data
knn_scores_sar = knn_cv_sar[dataset_use]
print('kNN SAR only - ' + dataset_use + ' :')
print(knn_scores_sar)

# Cross validate - kNN - OPT data
knn_scores_opt = knn_cv_opt[dataset_use]
print('kNN opt only - ' + dataset_use + ' :')
print(knn_scores_opt) 


# Cross validate - Random Forest - All data
rf_scores_all = rf_cv_all[dataset_use]
print('RF OPT+SAR - ' + dataset_use + ' :')
print(rf_scores_all) 

# Cross validate - Random Forest - SAR data
rf_scores_sar = rf_cv_sar[dataset_use]
print('RF OPT+SAR - ' + dataset_use + ' :')
print(rf_scores_sar)

# Cross validate - Random Forest - OPT data
rf_scores_opt = rf_cv_opt[dataset_use]
print('RF OPT - ' + dataset_use + ' :')
print(rf_scores_opt)         

