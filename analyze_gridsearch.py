#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:49:55 2020

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

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

#%% Input files and plot options
# Plot cross-set results?
do_cross_set = False
plot_dataset_rf = False
plot_dataset_knn = False

# Input file
gridsearch_file = 'gridsearch_pgnlm_20200103.pkl' #'gridsearch_C3features_20200109-threeclass.pkl' #'gridsearch_C3_20200108-threeclass.pkl' #'gridsearch_C3_20200108-twoclass.pkl' # 'gridsearch_pgnlm_20200107-twoclass.pkl' # 'gridsearch_pgnlm_20200106-5fold.pkl' #                        
# Prefix for object filename
datamod_fprefix = 'cov_mat-20200108.pkl' #'20191220_PGNLM-paramsearch'       
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix

#%% Read DataModalities object with ground in situ vegetation data
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


#%% Read result file
with open(os.path.join(dirname, 'data', gridsearch_file), 'rb') as infile:
        result_summary, param_list, result_rf_cross_val, result_knn_cross_val,\
        result_svm_cross_val, result_rf_cross_set, result_knn_cross_set,\
        result_svm_cross_set = pickle.load(infile)

#%% Read values from list of dicts into lists
xval_knn_max = [d['cross-val_knn_max'] for d in result_summary]
xval_rf_max = [d['cross-val_rf_max'] for d in result_summary]
xval_svm_max = [d['cross-val_svm_max'] for d in result_summary]
largest_class_size = [d['largest_class_size'] for d in result_summary]

# 20200110 - convert to numpy array, since it's considerably easier to work with
largest_class_size = np.asarray(largest_class_size)

if do_cross_set: 
    xset_knn_max = [d['cross-set_knn_max'] for d in result_summary]
    xset_rf_max = [d['cross-set_rf_max'] for d in result_summary]
    xset_svm_max = [d['cross-set_svm_max'] for d in result_summary]

#%% Calculate Gain
best_knn_xval_gain = np.asarray(xval_knn_max) - largest_class_size
best_rf_xval_gain = np.asarray(xval_rf_max) - largest_class_size
best_svm_xval_gain = np.asarray(xval_svm_max) - largest_class_size
if do_cross_set: 
    best_knn_xset_gain = np.asarray(xset_knn_max) - largest_class_size
    best_rf_xset_gain = np.asarray(xset_rf_max) - largest_class_size
    best_svm_xset_gain = np.asarray(xset_svm_max) - largest_class_size


#%% Find unique values of largest class size
lcs_unique, lcs_indice, lcs_inverse, lcs_counts = np.unique(largest_class_size,
                    return_index=True, return_inverse=True, return_counts=True)

                                                            
# Which results to process
result_list = result_rf_cross_val
                                                            
# Intialize output result list
best_result_params = []
best_result_acc = np.zeros(lcs_unique.shape)

# loop over datasets here???
current_dataset = 'C-20200101-0608'
for lcs_index, lcs_size in enumerate(lcs_unique):
    # Find runs with parameters that result in this largest class size (lcs)
    curr_ind = np.where(lcs_inverse==lcs_index)
    # Loop over classification results that match the largest dataset size
    best_candidates_acc = -1
    for i_run in curr_ind[0]:
        # Get accuracy for dataset for current parameter set
        curr_accuracy = result_list[i_run][current_dataset]
        # Check if the result is better than the current best
        if curr_accuracy > best_candidates_acc:
            # New high score, make it new best candidate
            best_candidates_acc = curr_accuracy
            # Store parameters
            best_candidates_params = [param_list[i_run]]
        elif curr_accuracy == best_candidates_acc:
            # Add parameters to list of best parameters
            best_candidates_params.append(param_list[i_run])
            
    # Add to overall list of best parameters
    best_result_params += best_candidates_params
    best_result_acc[lcs_index] = best_candidates_acc

        
#%% Find parameters that gives the highest accuracy for each dataset

# Initialize output arrays
n_param_sets = length(best_result_params)
knn_k_vec = np.zeros(n_param_sets)

# Loop over result list
for i_params, parameters in enumerate(best_result_params):
    knn_k_vec[i_params] = parameters['knn_k'] # Makes no sense for RF results

# Plot histogram 
knn_k_bins = np.arange(1,11) # Low and high of randint for knn k

fig = plt.figure()
plt.hist(knn_k_vec, knn_k_bins, facecolor='g', alpha=0.5, label='knn k')
plt.legend(loc='upper right')
plt.ylabel('Counts')
plt.xlabel('k')
plt.title('Parameters with best accuracy results')

#%% Get list of datasets
dataset_keys = result_rf_cross_val[0].keys() # TODO: consider storing dataset keys in result_summary or other variable
dataset_list = list(dataset_keys)
n_sets = length(dataset_keys)
rf_result_vec = np.zeros((n_sets ,1))
rf_best_set_vec = np.zeros((n_sets ,1))
for i_run, result_dict in enumerate(result_rf_cross_val):
    # Get best result
    best_res = result_summary[i_run]['cross-val_rf_max']
    for i_dataset, key in enumerate(result_dict):
        # Find diff from best result
        rf_result_vec[i_dataset, 0] += result_dict[key] - best_res
        rf_best_set_vec[i_dataset, 0] += 1-np.ceil(best_res-result_dict[key])
            


#%% Summary of Random Forest (rf) results
if plot_dataset_rf:
    # Get list of datasets
    dataset_keys = result_rf_cross_val[0].keys() # TODO: consider storing dataset keys in result_summary or other variable
    dataset_list = list(dataset_keys)
    n_sets = length(dataset_keys)
    rf_result_vec = np.zeros((n_sets ,1))
    rf_best_set_vec = np.zeros((n_sets ,1))
    for i_run, result_dict in enumerate(result_rf_cross_val):
        # Get best result
        best_res = result_summary[i_run]['cross-val_rf_max']
        for i_dataset, key in enumerate(result_dict):
            # Find diff from best result
            rf_result_vec[i_dataset, 0] += result_dict[key] - best_res
            rf_best_set_vec[i_dataset, 0] += 1-np.ceil(best_res-result_dict[key])
    
    # Diff                  
    fig = plt.figure()
    plt.title('RF dataset result, diff from max accuracy')
    plt.scatter(np.arange(n_sets), rf_result_vec, c='r', marker='o')
    plt.xlabel('Dataset nr.')
    plt.show()
    # Count number of best results
    fig = plt.figure()
    plt.title('Number of best accuracy results, RF')
    plt.scatter(np.arange(n_sets), rf_best_set_vec, c='r', marker='o')
    plt.xlabel('Dataset nr.')
    plt.show()
    
    # Find best set
    knn_best_diff = dataset_list[np.argmax(rf_result_vec)]
    knn_best_vote = dataset_list[np.argmax(rf_best_set_vec)]
    print('kNN, best average deviation from max accuracy: ' + knn_best_diff)
    print('kNN, most max accuracy results: ' + knn_best_vote)
    # Result for all runs
    xval_knn_bestdiff = [d[knn_best_diff] for d in result_rf_cross_val]
    
#%% Summary of kNN results
if plot_dataset_knn:
    # Get list of datasets
    dataset_keys = result_knn_cross_val[0].keys() # TODO: consider storing dataset keys in result_summary or other variable
    n_sets = length(dataset_keys)
    knn_result_vec = np.zeros((n_sets ,1))
    for i_run, result_dict in enumerate(result_knn_cross_val):
        # Get best result
        best_res = result_summary[i_run]['cross-val_knn_max']
        for i_dataset, key in enumerate(result_dict):
            # Find diff from best result
            knn_result_vec[i_dataset, 0] += result_dict[key] - best_res
                      
    fig = plt.figure()
    plt.title('KNN dataset result, diff from max accuracy')
    plt.scatter(np.arange(n_sets), knn_result_vec, c='b', marker='o')
    plt.ylabel('Dataset nr.')
    plt.show()
    
if plot_dataset_rf and plot_dataset_knn:
    sum_result_vec = rf_result_vec+knn_result_vec
    
    fig = plt.figure()
    plt.title('RF+KNN dataset result, diff from max accuracy')
    plt.scatter(np.arange(n_sets), sum_result_vec, c='k', marker='o')
    plt.ylabel('Dataset nr.')
    plt.show()
    
    max_diff_avg = np.max(sum_result_vec)/n_sets
    print(list(dataset_keys)[np.argmax(sum_result_vec)])


#%% PLOT GAIN
fig = plt.figure()
plt.scatter(largest_class_size, best_knn_xval_gain, c='r', marker='o')
plt.scatter(largest_class_size, best_rf_xval_gain, c='b', marker='x')
plt.scatter(largest_class_size, best_svm_xval_gain, c='g', marker='+')
plt.xlabel('Largest class size'); plt.ylabel('Max accuracy gain')
#plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
plt.title('Gain - cross validation')
plt.legend(['KNN', 'RF', 'SVM'])
plt.show()

#%% PLOT OVERALL ACCURACY - cross validation
fig = plt.figure()
plt.scatter(largest_class_size, xval_knn_max, c='r')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
if plot_dataset_knn:
    plt.scatter(largest_class_size, xval_knn_bestdiff, c='b')
#plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((0,1)); plt.xlim((0,1))
#plt.legend(['other', 'Live', 'Defoliated'])
plt.title('KNN max accuracy - cross validation')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xval_knn_max, c='b')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
plt.ylim((0,1)); plt.xlim((0,1))
plt.title('RF max accuracy - cross validation')
plt.show()

fig = plt.figure()
plt.scatter(largest_class_size, xval_svm_max, c='g')
plt.plot([0,1], [0,1], c='g')
plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
plt.ylim((0,1)); plt.xlim((0,1))
plt.title('SVM max accuracy - cross validation')
plt.show()

#%% Do plots for cross-set classification
if do_cross_set: 
    fig = plt.figure()
    plt.scatter(largest_class_size, best_knn_xset_gain, c='r', marker='o')
    plt.scatter(largest_class_size, best_rf_xset_gain, c='b', marker='x')
    plt.scatter(largest_class_size, best_svm_xset_gain, c='g', marker='+')
    plt.xlabel('Largest class size'); plt.ylabel('Max accuracy gain')
    plt.title('Gain - Train on one set, test on others')
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
    plt.ylim((0,1)); plt.xlim((0,1))
    plt.title('RF max accuracy - Train on one set, test on others')
    plt.show()
    
    fig = plt.figure()
    plt.scatter(largest_class_size, xset_svm_max, c='g')
    plt.plot([0,1], [0,1], c='g')
    plt.xlabel('Largest class size'); plt.ylabel('Accuracy')
    plt.ylim((0,1)); plt.xlim((0,1))
    plt.title('SVM max accuracy - Train on one set, test on others')
    plt.show()

