#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:16:49 2019

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import os # Necessary for relative paths
import pickle # To load object
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Input file
gridsearch_file = 'gridsearch_1.pkl'

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'
                         
# Prefix for object filename
datamod_fprefix = 'All-data-0919'
          
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + '-' + '.pkl'

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)

# Read ground truth point measurements into a matrix 
y_var_read = ['plc', 'pdc', 'n_trees']
n_obs_y = length(all_data.idx_list) # Number of observations
n_var_y = length(y_var_read) # Number of ecological variables read 
y_data = np.empty((n_obs_y, n_var_y))
# Loop through list of variables and add to Y mat out
for i_var_y in range(n_var_y):
    y = all_data.read_data_points(y_var_read[i_var_y])
    # Ensure that the data has the correct format and remove NaNs
    y = y.astype(float)
    y[np.isnan(y)] = 0 # Replace NaNs with zeros
    y_data[:,i_var_y] = y

# Read or create result dicts - kNN

# Read predefined file
with open(os.path.join(dirname, 'data', gridsearch_file), 'rb') as infile:
    result_summary, param_list, result_rf_cross_val, result_knn_cross_val, result_rf_cross_set, result_knn_cross_set = pickle.load(infile)

## Initialize output lists
#param_list = []
##result_rf_kappa = []
##result_knn_kappa = []
#result_summary = []
#result_rf_cross_val = []
#result_rf_cross_set = []
#result_knn_cross_val = []
#result_knn_cross_set = []

# Read values from list of dicts into lists
best_knn_gain = [d['best_knn_gain'] for d in result_summary]
best_rf_gain = [d['best_rf_gain'] for d in result_summary]
largest_class_size = [d['largest_class_size'] for d in result_summary]

# Total accuracy
best_knn_acc = np.asarray(best_knn_gain) + np.asarray(largest_class_size)

fig = plt.figure()
plt.scatter(largest_class_size, best_knn_gain)
plt.xlabel('Largest class size'); plt.ylabel('Max accuracy gain')
#plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
#plt.legend(['other', 'Live', 'Defoliated'])
#plt.legend()
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, best_knn_acc)
plt.plot([0,1], [0,1], c='r')
plt.xlabel('Largest class size'); plt.ylabel('Max accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
#plt.legend()
plt.show()


