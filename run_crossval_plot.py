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

# kNN output lists (9)
knn_all_means = []
knn_all_stds = []
knn_all_idx = []
knn_sar_means = []
knn_sar_stds = []
knn_sar_idx = []
knn_opt_means = []
knn_opt_stds = []
knn_opt_idx = []

# rf output lists (9)
rf_all_means = []
rf_all_stds = []
rf_all_idx = []
rf_sar_means = []
rf_sar_stds = []
rf_sar_idx = []
rf_opt_means = []
rf_opt_stds = []
rf_opt_idx = []

# Index
idx_data = 0

# Loop through all satellite images
for dataset_use in dataset_list:
    idx_data += 2
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
    # Add to list
    knn_sar_means.append(np.mean(knn_scores_sar))
    knn_sar_stds.append(np.std(knn_scores_sar))
    knn_sar_idx.append(idx_data+0.5)
    
    # Cross validate - kNN - OPT data
    knn_scores_opt = knn_cv_opt[dataset_use]
    print('kNN opt only - ' + dataset_use + ' :')
    print(knn_scores_opt)
    # Add to list
    knn_opt_means.append(np.mean(knn_scores_opt))
    knn_opt_stds.append(np.std(knn_scores_opt))
    knn_opt_idx.append(idx_data-0.5)
    
    
    # Cross validate - RF - All data
    rf_scores_all = rf_cv_all[dataset_use]
    print('rf OPT+SAR - ' + dataset_use + ' :')
    print(rf_scores_all) 
    # Add to list
    rf_all_means.append(np.mean(rf_scores_all))
    rf_all_stds.append(np.std(rf_scores_all))
    rf_all_idx.append(idx_data)
    
    # Cross validate - RF - SAR data
    rf_scores_sar = rf_cv_sar[dataset_use]
    print('rf SAR only - ' + dataset_use + ' :')
    print(rf_scores_sar)
    # Add to list
    rf_sar_means.append(np.mean(rf_scores_sar))
    rf_sar_stds.append(np.std(rf_scores_sar))
    rf_sar_idx.append(idx_data+0.5)
    
    # Cross validate - RF - OPT data
    rf_scores_opt = rf_cv_opt[dataset_use]
    print('rf opt only - ' + dataset_use + ' :')
    print(rf_scores_opt)
    # Add to list
    rf_opt_means.append(np.mean(rf_scores_opt))
    rf_opt_stds.append(np.std(rf_scores_opt))
    rf_opt_idx.append(idx_data-0.5)
     

# kNN cross-validation accuracy
fig = plt.figure()
plt.xticks(knn_all_idx, dataset_list)
plt.errorbar(knn_all_idx, knn_all_means, knn_all_stds, linestyle='None', marker='^', capsize=3)
plt.errorbar(knn_sar_idx, knn_sar_means, knn_sar_stds, linestyle='None', marker='^', capsize=3)
plt.errorbar(knn_opt_idx, knn_opt_means, knn_opt_stds, linestyle='None', marker='^', capsize=3)
axes = plt.gca()
axes.set_xlim([np.min(knn_all_idx)-1.0, np.max(knn_all_idx)+1.0])
axes.set_ylim([0,1])
plt.title('kNN cross-validation accuracy')
plt.show()


# RF cross-validation accuracy
fig = plt.figure()
plt.xticks(rf_all_idx, dataset_list)
plt.errorbar(rf_all_idx, rf_all_means, rf_all_stds, linestyle='None', marker='^', capsize=3)
plt.errorbar(rf_sar_idx, rf_sar_means, rf_sar_stds, linestyle='None', marker='^', capsize=3)
plt.errorbar(rf_opt_idx, rf_opt_means, rf_opt_stds, linestyle='None', marker='^', capsize=3)
axes = plt.gca()
axes.set_xlim([np.min(rf_all_idx)-1.0, np.max(rf_all_idx)+1.0])
axes.set_ylim([0,1])
plt.title('RF cross-validation accuracy')
plt.show()
