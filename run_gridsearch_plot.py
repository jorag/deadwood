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
gridsearch_file = 'gridsearch_20191205.pkl' # 'gridsearch_DiffGPS.pkl'

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'
                         
# Prefix for object filename
datamod_fprefix = 'New-data-20191205' # 'All-data-0919'
          
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

# Read result file
with open(os.path.join(dirname, 'data', gridsearch_file), 'rb') as infile:
        result_summary, param_list, result_rf_cross_val, result_knn_cross_val,\
        result_svm_cross_val, result_rf_cross_set, result_knn_cross_set,\
        result_svm_cross_set = pickle.load(infile)

# Read values from list of dicts into lists
xset_knn_max = [d['cross-set_knn_max'] for d in result_summary]
xset_rf_max = [d['cross-set_rf_max'] for d in result_summary]
xset_svm_max = [d['cross-set_svm_max'] for d in result_summary]
xval_knn_max = [d['cross-val_knn_max'] for d in result_summary]
xval_rf_max = [d['cross-val_rf_max'] for d in result_summary]
xval_svm_max = [d['cross-val_svm_max'] for d in result_summary]
largest_class_size = [d['largest_class_size'] for d in result_summary]

# Gain
best_knn_xset_gain = np.asarray(xset_knn_max) - np.asarray(largest_class_size)
best_rf_xset_gain = np.asarray(xset_rf_max) - np.asarray(largest_class_size)
best_svm_xset_gain = np.asarray(xset_svm_max) - np.asarray(largest_class_size)
best_knn_xval_gain = np.asarray(xval_knn_max) - np.asarray(largest_class_size)
best_rf_xval_gain = np.asarray(xval_rf_max) - np.asarray(largest_class_size)
best_svm_xval_gain = np.asarray(xval_svm_max) - np.asarray(largest_class_size)


# PLOT GAIN
fig = plt.figure()
plt.scatter(largest_class_size, best_knn_xset_gain, c='r', marker='o')
plt.scatter(largest_class_size, best_rf_xset_gain, c='b', marker='x')
plt.scatter(largest_class_size, best_svm_xset_gain, c='g', marker='+')
plt.xlabel('Largest class size'); plt.ylabel('Max accuracy gain')
plt.title('Gain - Train on one set, test on others')
plt.legend(['KNN', 'RF', 'SVM'])
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, best_knn_xset_gain, c='r', marker='o')
plt.scatter(largest_class_size, best_rf_xset_gain, c='b', marker='x')
plt.scatter(largest_class_size, best_svm_xset_gain, c='g', marker='+')
plt.xlabel('Largest class size'); plt.ylabel('Max accuracy gain')
#plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
plt.title('Gain - cross validation')
plt.legend(['KNN', 'RF', 'SVM'])
plt.show()

# PLOT OVERALL ACCURACY - cross set
fig = plt.figure()
plt.scatter(largest_class_size, xset_knn_max, c='r')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('KNN max accuracy - Train on one set, test on others')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xset_knn_max, c='b')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('RF max accuracy - Train on one set, test on others')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xset_svm_max, c='g')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('SVM max accuracy - Train on one set, test on others')
plt.show()

# PLOT OVERALL ACCURACY - cross validation
fig = plt.figure()
plt.scatter(largest_class_size, xval_knn_max, c='r')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('KNN max accuracy - cross validation')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xval_knn_max, c='b')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('RF max accuracy - cross validation')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xval_svm_max, c='g')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('SVM max accuracy - cross validation')
plt.show()


