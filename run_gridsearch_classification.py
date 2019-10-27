#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:48:05 2019

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
#from gdalconst import *
import tkinter
from tkinter import filedialog
import os # Necessary for relative paths
import pickle # To load object
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # train/test set split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # is deafault in cross-val?
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Output files
gridsearch_file = 'gridsearch_1.pkl'

# Parameters
n_runs = 3
crossval_split_k = 3
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k)
# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])
# Min and max size of classes
min_class_size = 0.07
max_class_size = 0.70
# Normalization options to try
norm_options =  ['local','global','none']

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'
                         
# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# Prefix for object filename
datamod_fprefix = 'All-data-0919'
id_list = ['A', 'B', 'C'] # First is used for training, next is used for testing
          
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

# Initialize output lists
param_list = []
#result_rf_kappa = []
#result_knn_kappa = []
result_rf_cross_val = []
result_rf_cross_set = []
result_knn_cross_val = []
result_knn_cross_set = []

for i_run in range(n_runs):
    # PROCESSING PARAMETERS
    knn_k = np.random.randint(1, high=11)
    rf_ntrees = np.random.randint(5, high=100) # Number of trees in the Random Forest algorithm
    # Normalization
    norm_type = np.random.choice(norm_options) # 'local' # 'global' # 'none' # 
    # Class boundaries
    min_p_live = np.random.uniform(low=0.0, high=0.4)
    min_p_defo = np.random.uniform(low=0.0, high=0.4)
    min_tree_live = np.random.randint(0, high=7)
    diff_live_defo = np.random.uniform(low=0.0, high=0.1)
    
    # Set labels
    data_labels = np.zeros((length(y_data)))
    for i_point in range(length(data_labels)):
        if y_data[i_point, 2] >= min_tree_live and (y_data[i_point, 0]>min_p_live or y_data[i_point, 1]>min_p_defo): 
             if y_data[i_point, 0] >= y_data[i_point, 1] - diff_live_defo:
                 data_labels[i_point] = 1
             else:
                 data_labels[i_point] = 2
                                
    class_dict=class_dict_in
    n_classes = length(class_dict)
    # Convert labels to numpy array
    labels = np.asarray(data_labels)
    # Find unique labels and counts
    u_labels, label_percent = np.unique(labels, return_counts=True)
    label_percent = label_percent/n_obs_y # Make percent/fractiob
    # Check that all classes are represented 
    if (length(u_labels) != n_classes or np.min(label_percent) < min_class_size or 
             np.max(label_percent) > max_class_size):
        print(labels_percent)
        continue 
             
    
    # Print number of instances for each class
    for key in class_dict.keys():
        val = class_dict[key]
        n_instances = length(labels[labels==val])
        print(str(val)+' '+key+' - points: '+str(n_instances))
    
    # Collect performance measures in dict
    #rf_mean_kappa = dict()
    #knn_mean_kappa = dict()
    rf_acc = dict()
    knn_acc = dict()
    rf_mean_acc = dict()
    knn_mean_acc = dict()
    
    # Print number of instances for each class
    n_class_samples = []
    for key in class_dict.keys():
        val = class_dict[key]
        n_instances = length(labels[labels==val])
        n_class_samples.append(n_instances)
        print(str(val)+' '+key+' - points: '+str(n_instances))
    
    
    # TRAIN AND CROSS-VALIDATE
    prev_type = 'dummy'
    # Go through all satellite images and all data modalities in object
    for dataset_type in all_data.all_modalities:
        for dataset_id in id_list:           
            # Get satellite data
            try:
                dataset_use = dataset_type+'-'+dataset_id
                sat_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
                print('----------------------------------------------------------')
                print(dataset_use)
                curr_type = dataset_type # Dataset loaded ok
            except:
                continue
            
            # Ensure that the output array has the proper shape (2 dimensions)
            if length(sat_data.shape) == 1:
                # If data is only a single column make it a proper vector
                sat_data = sat_data[:, np.newaxis]
            elif length(sat_data.shape) > 2:
                # Remove singelton dimensions
                sat_data = np.squeeze(sat_data)
            
            # Do normalization
            sat_data = norm01(sat_data, norm_type=norm_type)
            # Name of input object and file with satellite data path string
            sat_pathfile_name = dataset_use + '-path'
            # Max values of data
            print('Max values', np.max(sat_data,axis=0))
            
            # Split into training and test datasets
            if prev_type != curr_type: # New data type, do training
            #data_train, data_test, labels_train, labels_test = train_test_split(sat_data, data_labels, test_size=test_pct, random_state=rnd_state)  
                #data_train = np.concatenate(data_train, sat_data)
                #labels_train = np.concatenate(labels_train, data_labels)
                # Fit kNN
                neigh = KNeighborsClassifier(n_neighbors=knn_k)
                neigh.fit(sat_data, data_labels)
                # Fit RF
                rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
                rf_all.fit(sat_data, data_labels)
            else: # Have one instance of the dataset already, Do testing 
                # Score kNN
                knn_score = neigh.score(sat_data, data_labels)
                knn_acc[dataset_use] = knn_score
                # Use kNN classifier
                print('------ kNN - ' + dataset_use + ' --------')
                print('Accuracy = ', knn_score) 
                # Test kNN on test dataset
                knn_prediction_result = neigh.predict(sat_data)
                # Print kNN confusion matrix
                knn_confmat = confusion_matrix(data_labels, knn_prediction_result)
                print('Confusion matrix:')
                print(knn_confmat)
                
                # Score Random Forest - All data
                rf_scores_all = rf_all.score(sat_data, data_labels)
                # Add to output dict
                rf_acc[dataset_use] = rf_scores_all
                # Use RF classifier
                print('------- Random Forest - ' + dataset_use + ' -------')
                print('Accuracy, = ', rf_scores_all)
                # Test RF on test dataset
                rf_prediction_result = rf_all.predict(sat_data)
                # Print RF confusion matrix
                rf_confmat = confusion_matrix(data_labels, rf_prediction_result)
                print('Confusion matrix:')
                print(rf_confmat)
            
            # Cross validate - kNN - All data
            knn_cv = KNeighborsClassifier(n_neighbors=knn_k)
            knn_scores_cv = cross_val_score(knn_cv, sat_data, data_labels, cv=crossval_kfold)
            # Add to output dict
            print('kNN - ' + dataset_use + ' :')
            print(np.mean(knn_scores_cv)) 
            knn_mean_acc[dataset_use] = np.mean(knn_scores_cv)
            
            # Cross validate - Random Forest - All data
            rf_cv = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
            rf_scores_cv = cross_val_score(rf_cv, sat_data, data_labels, cv=crossval_kfold)
            # Add to output dict
            print('Random Forest - ' + dataset_use + ' :')
            print(np.mean(rf_scores_cv)) 
            rf_mean_acc[dataset_use] = np.mean(rf_scores_cv)
            
            #data_train = []
            #labels_train = []
            # Set previous dataset type    
            prev_type = dataset_type
    
    # Add parameters to output dict            
    param_dict = dict()
    param_dict['knn_k'] = knn_k
    param_dict['rf_ntrees'] = rf_ntrees
    param_dict['norm_type'] = norm_type 
    param_dict['min_p_live'] = min_p_live
    param_dict['min_p_defo'] = min_p_defo
    param_dict['min_tree_live'] = min_tree_live
    param_dict['diff_live_defo'] = diff_live_defo
    # Add to output result
    param_list.append(param_dict)
    #result_rf_kappa = []
    #result_knn_kappa = []
    result_rf_cross_val.append(rf_mean_acc)
    result_rf_cross_set.append(rf_acc)
    result_knn_cross_val.append(knn_mean_acc)
    result_knn_cross_set.append(knn_acc)          

# SAVE RESULTS
# kNN - cross validation
with open(os.path.join(dirname, 'data', gridsearch_file), 'wb') as output:
    pickle.dump([param_list, result_rf_cross_val, result_knn_cross_val, result_rf_cross_set, result_knn_cross_set], output, pickle.HIGHEST_PROTOCOL)