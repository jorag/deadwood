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

# Prefix for input datamodalities object filename
datamod_fprefix = ''
# Prefix for input cross validation object filename
crossval_fprefix = ''

# Name of input object and file with satellite data path string
#dataset_use = 'vanZyl-B'
#obj_in_name = datamod_fprefix + dataset_use + '.pkl'
#sat_pathfile_name = dataset_use + '-path'

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

knn_file = datamod_fprefix + crossval_fprefix + 'cross_validation_knn.pkl'
rf_file = datamod_fprefix + crossval_fprefix + 'cross_validation_rf.pkl'
                          
# Read result dicts - kNN
# Read predefined file
with open(os.path.join(dirname, 'data', knn_file ), 'rb') as infile:
    knn_cv_all, knn_cv_sar, knn_cv_opt = pickle.load(infile)
    
# Read or create reult dicts - Random Forest
# Read predefined file
with open(os.path.join(dirname, 'data', rf_file ), 'rb') as infile:
    rf_cv_all, rf_cv_sar, rf_cv_opt = pickle.load(infile)
    
# List of datasets to process
dataset_list = ['Coh-A', 'Coh-B', 'Coh-C', 'vanZyl-A', 'vanZyl-B', 'vanZyl-C']
# dataset_list = ['vanZyl-A']

knn_all_means = []
knn_all_stds = []
knn_all_idx = []
idx_data = 0

# Loop through all satellite images
for dataset_use in dataset_list:
    idx_data += 1
    # GET CROSS-VALIDATE RESULTS
    
    # Cross validate - kNN - All data
    knn_scores_all = knn_cv_all[dataset_use]
    print('kNN OPT+SAR - ' + dataset_use + ' :')
    print(knn_scores_all) 
    # Add to list
    knn_all_means.append(np.mean(knn_scores_all))
    knn_all_stds.append(np.std(knn_scores_all))
    knn_all_idx.append(idx_data)
    
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
     

fig = plt.figure()
plt.errorbar(knn_all_idx, knn_all_means, knn_all_stds, linestyle='None', marker='^', capsize=3)
plt.show()
