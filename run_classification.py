# -*- coding: utf-8 -*-
"""
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # is deafault in cross-val?
from sklearn.metrics import confusion_matrix
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# PROCESSING PARAMETERS
crossval_split_k = 5
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k)
knn_k = 5
rf_ntrees = 10

# Image plot parameter - TODO: Differentiate for SAR and optical data
norm_type = 'local' # 'global' # 

# List of datasets to process
dataset_list = ['Coh-A', 'Coh-B', 'Coh-C', 'vanZyl-A', 'vanZyl-B', 'vanZyl-C']
#dataset_list = ['vanZyl-A', 'vanZyl-B', 'vanZyl-C']

# Prefix for input datamodalities object filename
datamod_fprefix = 'rule3m2_lai_globloc'
# Prefix for output cross validation object filename
crossval_fprefix = 'kNN' + str(knn_k) + 'trees' + str(rf_ntrees)


# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])
n_classes = length(class_dict_in)

# Plot classifier result for entire image
plot_image_result = True

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'
# Output files
knn_file =  datamod_fprefix + crossval_fprefix + 'cross_validation_knn.pkl'
knn_confmat_file =  datamod_fprefix + crossval_fprefix + 'conf_mat_knn.pkl'
rf_file = datamod_fprefix + crossval_fprefix + 'cross_validation_rf.pkl'
rf_confmat_file =  datamod_fprefix + crossval_fprefix + 'conf_mat_rf.pkl'
# Parameter save file
classify_params_file = datamod_fprefix + crossval_fprefix + 'cross_validation_params.pkl' 

# Read or create result dicts - kNN
try:
    # Read predefined file
    with open(os.path.join(dirname, 'data', knn_file), 'rb') as infile:
        knn_cv_all, knn_cv_sar, knn_cv_opt = pickle.load(infile)
except:
    knn_cv_all = dict(); knn_cv_sar = dict(); knn_cv_opt = dict()

# Read or create result dicts - kNN
try:
    # Read predefined file
    with open(os.path.join(dirname, 'data', knn_confmat_file), 'rb') as infile:
        knn_confmat_all, knn_confmat_sar, knn_confmat_opt = pickle.load(infile)
        print('CONF MATS LOADED')
except:
    knn_confmat_all = dict(); knn_confmat_sar = dict(); knn_confmat_opt = dict()


# Read or create result dicts - Random Forest
try:
    # Read predefined file
    with open(os.path.join(dirname, 'data', rf_file), 'rb') as infile:
        rf_cv_all, rf_cv_sar, rf_cv_opt = pickle.load(infile)
except:
    rf_cv_all = dict(); rf_cv_sar = dict(); rf_cv_opt = dict()
    
try:
    # Read predefined file
    with open(os.path.join(dirname, 'data', rf_confmat_file), 'rb') as infile:
        rf_confmat_all, rf_confmat_sar, rf_confmat_opt = pickle.load(infile)
        print('CONF MATS LOADED')
except:
    rf_confmat_all = dict(); rf_confmat_sar = dict(); rf_confmat_opt = dict()

                          


# TRAIN AND CROSS-VALIDATE
# Loop through all satellite images
for dataset_use in dataset_list:
    
    # Name of input object and file with satellite data path string
    obj_in_name = datamod_fprefix + dataset_use + '.pkl'
    sat_pathfile_name = dataset_use + '-path'
    
    
    # Load DataModalities object
    with open(os.path.join(dirname, "data", obj_in_name), 'rb') as input:
        input_data = pickle.load(input)
        
    # Get labels and class_dict (in case None is input, one is created)
    labels_out, class_dict = input_data.assign_labels(class_dict=class_dict_in)
    
    # Get all data
    all_data, data_labels = input_data.read_data_array(['quad_pol', 'optical'], 'all') 
    # Get SAR data
    sar_data, data_labels = input_data.read_data_array(['quad_pol'], 'all') 
    # Get OPT data
    opt_data, data_labels = input_data.read_data_array(['optical'], 'all') 
    # Convert labels to numpy array
    labels = np.asarray(data_labels)
    
    # Print number of instances for each class
    for key in class_dict.keys():
        val = class_dict[key]
        n_instances = length(labels[labels==val])
        print(str(val)+' '+key+' - points: '+str(n_instances))
    
    
    # Normalize data - should probably be done when data is stored in object...
    print(np.max(all_data,axis=0))
    
    # Split into training and test datasets
    data_train, data_test, labels_train, labels_test = train_test_split(all_data, data_labels, test_size=0.2, random_state=0)  
    
    # Create kNN classifier
    neigh = KNeighborsClassifier(n_neighbors=knn_k)
    # Fit kNN
    neigh.fit(data_train, labels_train) 
    
    # Score kNN
    print(neigh.score(data_test, labels_test)) 
    # Test kNN on test dataset
    prediction_result = neigh.predict(data_test)
    
    # Cross validate - kNN - All data
    knn_all = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores_all = cross_val_score(knn_all, all_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    knn_cv_all[dataset_use] = knn_scores_all
    print('kNN OPT+SAR - ' + dataset_use + ' :')
    print(np.mean(knn_scores_all)) 

    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(all_data, labels)
    # Initialize output confusion matrix
    knn_all_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(all_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = all_data[train_index], all_data[test_index]
       # Fit classifier
       knn_all.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, knn_all.predict(X_test))
       # Add contribution to overall confusion matrix
       knn_all_confmat += conf_mat_temp
       
    # Add to output dict
    knn_confmat_all[dataset_use] = knn_all_confmat
    print(knn_all_confmat)
    
    
    # Cross validate - kNN - SAR data
    knn_sar = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores_sar = cross_val_score(knn_sar, sar_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    knn_cv_sar[dataset_use] = knn_scores_sar
    print('kNN SAR only - ' + dataset_use + ' :')
    print(np.mean(knn_scores_sar))
    
    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(sar_data, labels)
    # Initialize output confusion matrix
    knn_sar_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(sar_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = sar_data[train_index], sar_data[test_index]
       # Fit classifier
       knn_sar.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, knn_sar.predict(X_test))
       # Add contribution to overall confusion matrix
       knn_sar_confmat += conf_mat_temp
       
    # Add to output dict
    knn_confmat_sar[dataset_use] = knn_sar_confmat
    print(knn_sar_confmat)
    

    # Cross validate - kNN - OPT data
    knn_opt = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores_opt = cross_val_score(knn_opt, opt_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    knn_cv_opt[dataset_use] = knn_scores_opt
    print('kNN opt only - ' + dataset_use + ' :')
    print(np.mean(knn_scores_opt)) 
    
    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(opt_data, labels)
    # Initialize output confusion matrix
    knn_opt_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(opt_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = opt_data[train_index], opt_data[test_index]
       # Fit classifier
       knn_opt.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, knn_opt.predict(X_test))
       # Add contribution to overall confusion matrix
       knn_opt_confmat += conf_mat_temp
       
    # Add to output dict
    knn_confmat_opt[dataset_use] = knn_opt_confmat
    print(knn_opt_confmat)
    
    
    
    # Cross validate - Random Forest - All data
    rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
    rf_scores_all = cross_val_score(rf_all, all_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    rf_cv_all[dataset_use] = rf_scores_all
    print('RF OPT+SAR - ' + dataset_use + ' :')
    print(np.mean(rf_scores_all)) 
    
    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(all_data, labels)
    # Initialize output confusion matrix
    rf_all_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(all_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = all_data[train_index], all_data[test_index]
       # Fit classifier
       rf_all.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, rf_all.predict(X_test))
       # Add contribution to overall confusion matrix
       rf_all_confmat += conf_mat_temp
       
    # Add to output dict
    rf_confmat_all[dataset_use] = rf_all_confmat
    print(rf_all_confmat)
    
    # Cross validate - Random Forest - SAR data
    rf_sar = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
    rf_scores_sar = cross_val_score(rf_sar, sar_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    rf_cv_sar[dataset_use] = rf_scores_sar
    print('RF SAR - ' + dataset_use + ' :')
    print(np.mean(rf_scores_sar))
    
    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(sar_data, labels)
    # Initialize output confusion matrix
    rf_sar_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(sar_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = sar_data[train_index], sar_data[test_index]
       # Fit classifier
       rf_sar.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, rf_sar.predict(X_test))
       # Add contribution to overall confusion matrix
       rf_sar_confmat += conf_mat_temp
       
    # Add to output dict
    rf_confmat_sar[dataset_use] = rf_sar_confmat
    print(rf_sar_confmat)
    
    
    # Cross validate - Random Forest - OPT data
    rf_opt = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
    rf_scores_opt = cross_val_score(rf_opt, opt_data, data_labels, cv=crossval_kfold)
    # Add to output dict
    rf_cv_opt[dataset_use] = rf_scores_opt
    print('RF OPT - ' + dataset_use + ' :')
    print(np.mean(rf_scores_opt))

    # Get split for cofusion matrix calculation
    skf = StratifiedKFold(n_splits=crossval_split_k)
    skf.get_n_splits(opt_data, labels)
    # Initialize output confusion matrix
    rf_opt_confmat = np.zeros((n_classes , n_classes ))
    # Use split
    for train_index, test_index in skf.split(opt_data, labels):
       # Split into training and test set
       y_train, y_test = labels[train_index], labels[test_index]
       X_train, X_test = opt_data[train_index], opt_data[test_index]
       # Fit classifier
       rf_opt.fit(X_train, y_train)
       # Calculate confusion matrix
       conf_mat_temp = confusion_matrix(y_test, rf_opt.predict(X_test))
       # Add contribution to overall confusion matrix
       rf_opt_confmat += conf_mat_temp
       
    # Add to output dict
    rf_confmat_opt[dataset_use] = rf_opt_confmat
    print(rf_opt_confmat)         


# SAVE RESULTS
# kNN - cross validation
with open(os.path.join(dirname, 'data', knn_file), 'wb') as output:
    pickle.dump([knn_cv_all, knn_cv_sar, knn_cv_opt], output, pickle.HIGHEST_PROTOCOL)
# kNN - confusion matrices   
with open(os.path.join(dirname, 'data', knn_confmat_file), 'wb') as output:
    pickle.dump([knn_confmat_all, knn_confmat_sar, knn_confmat_opt], output, pickle.HIGHEST_PROTOCOL)
# RF
with open(os.path.join(dirname, 'data', rf_file), 'wb') as output:
    pickle.dump([rf_cv_all, rf_cv_sar, rf_cv_opt], output, pickle.HIGHEST_PROTOCOL)

# Save parameters
with open(os.path.join(dirname, 'data', classify_params_file), 'wb') as output:
    pickle.dump([knn_k, rf_ntrees], output, pickle.HIGHEST_PROTOCOL)


# TEST ON COMPLETE IMAGE
if plot_image_result:
    ## Read satellite data
    try:
        # Read predefined file
        with open(os.path.join(dirname, 'data', sat_pathfile_name)) as infile:
            sat_file = infile.readline().strip()
            logit('Read file: ' + sat_file, log_type = 'default')
        
        # Load data
        dataset = gdal.Open(sat_file)
        gdalinfo_log(dataset, log_type='default')
    except:
        logit('Error, promt user for file.', log_type = 'default')
        # Predefined file failed for some reason, promt user
        root = tkinter.Tk() # GUI for file selection
        root.withdraw()
        sat_file = tkinter.filedialog.askopenfilename(title='Select input .tif file')
        # Load data
        dataset = gdal.Open(sat_file)
        gdalinfo_log(dataset, log_type='default')
    
    # Read multiple bands
    all_sat_bands = dataset.ReadAsArray()
    
    # Get bands used in two lists
    bands_use_single = []
    for key in input_data.modality_order: # .modality_bands.keys()
        # Get list of lists of bands
        bands_use_single += input_data.modality_bands[key]
    
    
    ## Predict class for entire satellite image
    sat_im = all_sat_bands[bands_use_single , :, : ]
    ## Reshape array to n_cols*n_rows rows with the channels as columns 
    sat_im_prediction, n_rows, n_cols = imtensor2array(sat_im)
    
    # NORMALIZE IMAGE - TODO: Change to 'local' both here and in get_sat_data??
    sat_im_prediction = norm01(sat_im_prediction, norm_type=norm_type, log_type = 'print')
    
    # For colourbar: Get the list of unique class numbers  
    class_n_unique = np.unique(list(class_dict.values()))
    # Use the highest and lowest class n for colourbar visualization
    class_n_lowest = np.min(class_n_unique) 
    class_n_highest = np.max(class_n_unique)
     
    colors = ['red','green','blue','purple']
    cmap = plt.get_cmap('jet', length(class_n_unique)) # Number of colours = n. of classes
    
                       
    # kNN image
    kNN_im_result = neigh.predict(sat_im_prediction)
    # Reshape to original input size
    sat_result_kNN = np.reshape(kNN_im_result, (n_rows, n_cols))
    
    # Show classification result
    fig = plt.figure()
    plt.imshow(sat_result_kNN.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
    plt.colorbar(ticks=np.unique(list(class_dict.values())) )
    plt.title('kNN result' + dataset_use)
    plt.show()  # display it
    
    
    # RF image
    rf_im_result = rf_all.predict(sat_im_prediction)
    # Reshape to original input size
    sat_result_rf = np.reshape(rf_im_result, (n_rows, n_cols))
    
    fig2 = plt.figure()
    plt.imshow(sat_result_rf.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
    plt.colorbar(ticks=np.unique(list(class_dict.values())) )
    plt.title('Random Forest result, '+dataset_use)
    plt.show()  # display it


