#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:47:54 2019

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import os # Necessary for relative paths
import pickle # To load object
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import cohen_kappa_score
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
# Single-run test parameters:
rnd_state = 33
    
# List of plots: # ['acc_bar_separate', 'acc_bar_combined', 'kappa_bar_combined', 'n_trees', 'plc_pdc_class'] 
plot_list = ['acc_bar_combined', 'plc_pdc_class'] 

# Prefix for object filename
datamod_fprefix = 'All-data-0919'
id_list = ['A', 'B', 'C'] # First is used for training, next is used for testing

# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + '-' + '.pkl'

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)

# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])

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

# Normalization
norm_type =  'local' # 'global' # 'none' # 

# Set labels
min_p_live = 0.10
min_p_defo = 0.05
min_tree_live = 2
diff_live_defo = 0.0
data_labels = np.zeros((length(y_data)))
for i_point in range(length(data_labels)):
    if y_data[i_point, 2] >= min_tree_live and (y_data[i_point, 0]>min_p_live or y_data[i_point, 1]>min_p_defo): 
         if y_data[i_point, 0] >= y_data[i_point, 1] - diff_live_defo:
             data_labels[i_point] = 1
         else:
             data_labels[i_point] = 2
                            
        
#data_labels[np.where(y_data[:,1]>0)] = 2 # Defoliated
#data_labels[np.where(y_data[:,1]<y_data[:,0])] = 1 # Live
#data_labels[np.where(y_data[:,1]<=0.015)] = 0 # Other
#data_labels[np.where(y_data[:,0]<=0.015)] = 0 # Other
#data_labels[np.where(y_data[:,2]<=1)] = 0 # Other
#data_labels = np.random.randint(3, size=(length(y_data)))
class_dict=class_dict_in
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
prev_type = 'dummy'
#data_train = []
#labels_train = []
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
            #data_train = np.vstack((data_train, sat_data))
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
            knn_mean_acc[dataset_use] = knn_score
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
            rf_mean_acc[dataset_use] = rf_scores_all
            # Use RF classifier
            print('------- Random Forest - ' + dataset_use + ' -------')
            print('Accuracy, = ', rf_scores_all)
            # Test RF on test dataset
            rf_prediction_result = rf_all.predict(sat_data)
            # Print RF confusion matrix
            rf_confmat = confusion_matrix(data_labels, rf_prediction_result)
            print('Confusion matrix:')
            print(rf_confmat)
        
        #data_train = []
        #labels_train = []
        # Set previous dataset type    
        prev_type = dataset_type
            
            

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
n_datasets = length(knn_mean_acc)
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
    plt.ylabel('Mean accuracy'); plt.ylim((0,1))
    #plt.legend(['Largest class %', 'RF', 'kNN'])
    plt.show()

if 'kappa_bar_combined' in plot_list:
    # Mean Kappa - both in same plot (kNN and RF)
    plt.figure()
    plt.bar(x_bars*2+ofs, list(rf_mean_kappa.values()), align='center', color='b', alpha=alf)
    plt.bar(x_bars*2-ofs, list(knn_mean_kappa.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars*2, list(rf_mean_kappa.keys()))
    plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k))
    plt.ylabel(r'Mean $\kappa$') 
    plt.legend(['RF', 'kNN'])
    plt.show()

if 'acc_bar_separate' in plot_list: 
    # Mean Accuracy - Separate plots (kNN and RF)
    # RF
    plt.figure()
    plt.bar(x_bars, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.xticks(x_bars, list(rf_mean_acc.keys())); plt.title('RF, n_trees: '+str(rf_ntrees))
    plt.ylabel('Mean accuracy'); plt.ylim((0,1))
    plt.show()
    # kNN
    plt.figure()
    plt.bar(x_bars, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars, list(knn_mean_acc.keys())); plt.title('kNN, k: '+str(knn_k))
    plt.ylabel('Mean accuracy'); plt.ylim((0,1))
    plt.show()