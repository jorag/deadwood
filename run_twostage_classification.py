#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:10:25 2019

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
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
from dataclass import *

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# PROCESSING PARAMETERS
knn_k = 5
rf_ntrees = 25 # Number of trees in the Random Forest algorithm
# Cross validation
crossval_split_k = 3
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k)
# Single-run test parameters:
test_pct = 0.25
rnd_state = 33
    
# List of plots: # ['acc_bar_separate', 'acc_bar_combined', 'kappa_bar_combined', 'n_trees', 'plc_pdc_class'] 
plot_list = ['acc_bar_combined', 'plc_pdc_class'] 

# Prefix for object filename
datamod_fprefix = 'All-data-0919'
id_list = ['A', 'B', 'C']

# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + '-' + '.pkl'

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)

# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_forest = dict([['other', 0], ['Forest', 1]])

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

# Set labels
data_labels = np.zeros((length(y_data))) # Other
data_labels[np.where(y_data[:,1] > 0.100)] = 1 # Forest
data_labels[np.where(y_data[:,0] > 0.075)] = 1 # Forest
data_labels[np.where(y_data[:,2]<=2)] = 0 # Other
class_dict=class_dict_forest
n_classes = length(class_dict)

# Convert labels to numpy array
labels = np.asarray(data_labels)
# Print number of instances for each class
for key in class_dict.keys():
    val = class_dict[key]
    n_instances = length(labels[labels==val])
    print(str(val)+' '+key+' - points: '+str(n_instances))

# Collect performance measures in dict
rf_mean_acc = dict()
knn_mean_acc = dict()
rf_mean_kappa = dict()
knn_mean_kappa = dict()


if 'plc_pdc_class' in plot_list:
    # Show PLC and PDC with class colours
    # TODO: 3D plot with n_trees as one axis
    plot_labels = labels.astype(int)
    # Get standard colour/plotstyle vector
    c_vec = mycolourvec()
    # Convert to numpy array for indexation
    c_vec = np.asarray(c_vec)
    fig = plt.figure()
    plt.scatter(y_data[:,0], y_data[:,1], c=c_vec[plot_labels], marker='o', label=plot_labels)
    plt.xlabel('Live Crown Proportion'); plt.ylabel('Defoliated Crown Proportion')
    plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
    #plt.legend(['other', 'Live', 'Defoliated'])
    #plt.legend()
    plt.show()

if 'n_trees' in plot_list:
    # Plot number of trees
    #xs = np.arange(length(y_data)) # range(n_datasets)
    fig = plt.figure()
    plt.plot(y_data[:,2], 'ro-')
    plt.show()

# TRAIN AND CROSS-VALIDATE
# Go through all satellite images and all data modalities in object
for dataset_id in id_list: 
    for dataset_type in all_data.all_modalities:           
        # Get satellite data
        try:
            dataset_use = dataset_type+'-'+dataset_id
            sat_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
            print('----------------------------------------------------------')
            print(dataset_use)
        except:
            continue
        
        # Ensure that the output array has the proper shape (2 dimensions)
        if length(sat_data.shape) == 1:
            # If data is only a single column make it a proper vector
            sat_data = sat_data[:, np.newaxis]
        elif length(sat_data.shape) > 2:
            # Remove singelton dimensions
            sat_data = np.squeeze(sat_data)
        
        # Name of input object and file with satellite data path string
        sat_pathfile_name = dataset_use + '-path'
        # Max values of data
        print('Max values', np.max(sat_data,axis=0))
        
        # Split into training and test datasets
        data_train, data_test, labels_train, labels_test = train_test_split(sat_data, data_labels, test_size=test_pct, random_state=rnd_state)  
        
        
        # Create kNN classifier
        print('------ kNN - ' + dataset_use + ' --------')
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        # Fit kNN
        neigh.fit(data_train, labels_train) 
        
        # Score kNN
        print('Accuracy, test set fraction: '+str(test_pct)+' = ', neigh.score(data_test, labels_test)) 
        # Test kNN on test dataset
        prediction_result = neigh.predict(data_test)
        
        # Get training accuracy for entire dataset
        neigh2 = KNeighborsClassifier(n_neighbors=knn_k)
        # Fit kNN
        neigh2.fit(sat_data, data_labels) 
        
        # Score kNN
        print('Accuracy training set (max) = ', neigh2.score(sat_data, data_labels)) 
        # Test kNN on test dataset
        prediction_result2 = neigh2.predict(sat_data)
        print('n training errors = ', np.sum(np.abs(prediction_result2-data_labels)))
        
        # Cross validate - kNN - All data
        knn_all = KNeighborsClassifier(n_neighbors=knn_k)
        knn_scores_all = cross_val_score(knn_all, sat_data, data_labels, cv=crossval_kfold)
        # Add to output dict
        print('Accuracy, mean of '+str(crossval_split_k)+'-fold split= ', np.mean(knn_scores_all)) 
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, labels)
        # Initialize output confusion matrix and kappa
        knn_all_confmat = np.zeros((n_classes , n_classes))
        knn_all_kappa = []
        # Use split
        for train_index, test_index in skf.split(sat_data, labels):
           # Split into training and test set
           y_train, y_test = labels[train_index], labels[test_index]
           X_train, X_test = sat_data[train_index], sat_data[test_index]
           # Fit classifier
           knn_all.fit(X_train, y_train)
           # Do prediction
           y_pred = knn_all.predict(X_test)
           # Calculate confusion matrix
           conf_mat_temp = confusion_matrix(y_test, y_pred)
           # Add contribution to overall confusion matrix
           knn_all_confmat += conf_mat_temp
           # Calculate kappa
           knn_all_kappa.append(cohen_kappa_score(y_test, y_pred))
           
        # Add to output dict
        knn_mean_kappa[dataset_use] = np.mean(knn_all_kappa)
        print('kappa = ', np.mean(knn_all_kappa))
        print('Confusion matrix ('+str(crossval_split_k)+'-fold split):')
        print(knn_all_confmat)
        
        
        # Cross validate - Random Forest - All data
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, sat_data, data_labels, cv=crossval_kfold)
        # Add to output dict
        print('------- Random Forest - ' + dataset_use + ' -------')
        print('Accuracy, mean of '+str(crossval_split_k)+'-fold split= ', np.mean(rf_scores_all)) 
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, labels)
        # Initialize output confusion matrix
        rf_all_confmat = np.zeros((n_classes , n_classes))
        rf_all_kappa = []
        # Use split
        for train_index, test_index in skf.split(sat_data, labels):
           # Split into training and test set
           y_train, y_test = labels[train_index], labels[test_index]
           X_train, X_test = sat_data[train_index], sat_data[test_index]
           # Fit classifier
           rf_all.fit(X_train, y_train)
           # Do prediction
           y_pred = rf_all.predict(X_test)
           # Calculate confusion matrix
           conf_mat_temp = confusion_matrix(y_test, y_pred)
           # Add contribution to overall confusion matrix
           rf_all_confmat += conf_mat_temp
           # Calculate kappa
           rf_all_kappa.append(cohen_kappa_score(y_test, y_pred))
           
        # Add to output dict
        rf_mean_kappa[dataset_use] = np.mean(rf_all_kappa)
        print('kappa = ', np.mean(rf_all_kappa))
        print('Confusion matrix ('+str(crossval_split_k)+'-fold split):')
        print(rf_all_confmat)


# Convert labels to numpy array
labels = np.asarray(data_labels)
# Print number of instances for each class
n_class_samples = []
for key in class_dict.keys():
    val = class_dict[key]
    n_instances = length(labels[labels==val])
    n_class_samples.append(n_instances)
    print(str(val)+' '+key+' - points: '+str(n_instances))

# Plot summary statistics
n_datasets = length(rf_mean_acc)
x_bars = np.arange(n_datasets) # range(n_datasets)
ofs = 0.25 # offset
alf = 0.7 # alpha

# # Try sorting dictionaries alphabetically
#sorted(rf_mean_acc, key=rf_mean_acc.get, reverse=True)
#sorted(linreg_pdc_r2, key=linreg_pdc_r2.get, reverse=True)

if 'acc_bar_combined' in plot_list:
    # Mean Accuracy - both in same plot (kNN and RF)
    plt.figure()
    plt.bar(x_bars*2+ofs, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.bar(x_bars*2-ofs, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*n_datasets)
    plt.xticks(x_bars*2, list(rf_mean_acc.keys()))
    plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k))
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.legend(['Largest class %', 'RF', 'kNN'])
    plt.show()

if 'kappa_bar_combined' in plot_list:
    # Mean Kappa - both in same plot (kNN and RF)
    plt.figure()
    plt.bar(x_bars*2+ofs, list(rf_mean_kappa.values()), align='center', color='b', alpha=alf)
    plt.bar(x_bars*2-ofs, list(knn_mean_kappa.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars*2, list(rf_mean_kappa.keys()))
    plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k))
    plt.ylabel(r'Mean $\kappa$, n_splits = '+str(crossval_split_k)) 
    plt.legend(['RF', 'kNN'])
    plt.show()

if 'acc_bar_separate' in plot_list: 
    # Mean Accuracy - Separate plots (kNN and RF)
    # RF
    plt.figure()
    plt.bar(x_bars, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.xticks(x_bars, list(rf_mean_acc.keys())); plt.title('RF, n_trees: '+str(rf_ntrees))
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.show()
    # kNN
    plt.figure()
    plt.bar(x_bars, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars, list(knn_mean_acc.keys())); plt.title('kNN, k: '+str(knn_k))
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.show()
