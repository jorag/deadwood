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

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Prefix for object filename
datamod_fprefix = 'PGNLM-SNAP_C3_20200112' #'20191220_PGNLM-paramsearch' #'cov_mat-20200108' # 'New-data-20191203-' #'New-data-20191203-.pkl'
          
# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix  + '.pkl'
                          
#%% Classify LIVE FOREST vs. DEFOLIATED FOREST (and vs. OTHER?)
twoclass_only = True

# Normalization
norm_type = 'local' # 'global' # 'none' # 
# Class boundaries
min_p_live = 0.0250
min_p_defo = 0.0250 #0.060 
min_tree_live = 2
diff_live_defo = 0.025 #0 # 

#%% PROCESSING PARAMETERS
crossval_split_k = 3
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, shuffle=True, random_state=crossval_split_k)
knn_k = 5
rf_ntrees = 150 # Number of trees in the Random Forest algorithm

#%% Plot options
combined_bar_plots = False # Combination of RF and kNN result
separate_bar_plots = False # Separate of RF and kNN result
separate_dataset_plots = True # Separate plot for each dataset ID (A, B, C)
plot_class_boundaries = False
plot_kappa = False
plot_image_result = False 

# Prefix for output cross validation object filename
crossval_fprefix = 'new-kNN' + str(knn_k) + 'trees' + str(rf_ntrees)

#%% Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)

# List of dataset IDs, either a single letter or a letter + datestr combo
try:
    pgnlm_flag = True
    id_list = list(all_data.pgnlm_param_dict.keys()) 
except:
    pgnlm_flag = False
    id_list = ['A', 'B', 'C'] # First is used for training, next is used for testing

# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])

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
data_labels = np.zeros((length(y_data)))
for i_point in range(length(data_labels)):
    if y_data[i_point, 2] >= min_tree_live and (y_data[i_point, 0]>min_p_live or y_data[i_point, 1]>min_p_defo): 
         if y_data[i_point, 0] >= y_data[i_point, 1] - diff_live_defo:
             data_labels[i_point] = 1
         else:
             data_labels[i_point] = 2

#Number of classes
n_classes = length(class_dict)

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
        logit('CONF MATS LOADED')
except:
    rf_confmat_all = dict(); rf_confmat_sar = dict(); rf_confmat_opt = dict()


#%% Convert labels to numpy array
labels = np.array(data_labels)

# Remove "Other" class to test classification of "Live" vs. "Defoliated" 
if twoclass_only:
    data_labels = np.delete(labels, np.where(labels == 0))
    n_classes = 2
    n_samples_use = length(data_labels)
    print(n_samples_use)
else:
    # Number of classes
    n_classes = length(class_dict)

# Print number of instances for each class
for key in class_dict.keys():
    val = class_dict[key]
    n_instances = length(labels[labels==val])
    print(str(val)+' '+key+' - points: '+str(n_instances))

#%% Collect performance measures in dict
rf_mean_acc = dict()
knn_mean_acc = dict()
rf_mean_kappa = dict()
knn_mean_kappa = dict()

# Show PLC and PDC with class colours
# TODO: 3D plot with n_trees as one axis
plot_labels = labels.astype(int)
# Get standard colour/plotstyle vector
c_vec = mycolourvec()
# Convert to numpy array for indexation
c_vec = np.asarray(c_vec)

# Plot x
if plot_class_boundaries:
    xs = np.arange(length(y_data)) # range(n_datasets)
    fig = plt.figure()
    plt.scatter(y_data[:,0], y_data[:,1], c=c_vec[plot_labels], marker='o', label=plot_labels)
    plt.xlabel('Live Crown Proportion'); plt.ylabel('Defoliated Crown Proportion')
    plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
    plt.show()

#%% TRAIN AND CROSS-VALIDATE
for dataset_type in all_data.all_modalities: 
    # Check if PGNLM filtered or C3 matrix and select feature type
    if dataset_type.lower()[0:5] in ['pgnlm']:
        c3_feature_type = 'iq2c3'
    elif dataset_type.lower()[-2:] in ['c3']:
        c3_feature_type = 'c3snap_filtered'
    else:
        print('No feature type found for: '+dataset_type)
        c3_feature_type = 'NA'
        
    for dataset_id in id_list:           
        # Get satellite data
        try:
            if pgnlm_flag:
                dataset_use = dataset_id
            else:
                dataset_use = dataset_type+'-'+dataset_id
            sat_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
            curr_type = dataset_type # Dataset loaded ok
        except:
            print('Skipping: '+dataset_type+'-'+dataset_id)
            continue
        
        # Ensure that the output array has the proper shape (2 dimensions)
        if length(sat_data.shape) == 1:
            # If data is only a single column make it a proper vector
            sat_data = sat_data[:, np.newaxis]
        elif length(sat_data.shape) > 2:
            # Remove singelton dimensions
            sat_data = np.squeeze(sat_data)
        
         # Remove "Other" class to test classification of "Live" vs. "Defoliated" 
        if twoclass_only:
            sat_data = np.delete(sat_data, np.where(labels == 0), axis=0)
        
        # Name of input object and file with satellite data path string
        sat_pathfile_name = dataset_use + '-path'
        
        # Extract SAR covariance matrix features?
        sat_data = get_sar_features(sat_data, feature_type=c3_feature_type)
        
        # Normalize data
        print(np.max(sat_data,axis=0))
        # Do normalization
        sat_data = norm01(sat_data, norm_type=norm_type)
        print(np.max(sat_data,axis=0))
        
        # Split into training and test datasets
        data_train, data_test, labels_train, labels_test = train_test_split(sat_data, data_labels, test_size=0.2, random_state=0)  
        
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
        knn_scores_all = cross_val_score(knn_all, sat_data, data_labels, cv=crossval_kfold)
        # Add to output dict
        knn_cv_all[dataset_use] = knn_scores_all
        print('kNN - ' + dataset_use + ' :')
        print(np.mean(knn_scores_all)) 
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, data_labels)
        # Initialize output confusion matrix and kappa
        knn_all_confmat = np.zeros((n_classes , n_classes))
        knn_all_kappa = []
        # Use split
        for train_index, test_index in skf.split(sat_data, data_labels):
           # Split into training and test set
           y_train, y_test = data_labels[train_index], data_labels[test_index]
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
        knn_confmat_all[dataset_use] = knn_all_confmat
        knn_mean_kappa[dataset_use] = np.mean(knn_all_kappa)
        print(knn_all_confmat)
        print('kappa = ', np.mean(knn_all_kappa))
        
        
        # Cross validate - Random Forest - All data
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, sat_data, data_labels, cv=crossval_kfold)
        # Add to output dict
        rf_cv_all[dataset_use] = rf_scores_all
        print('Random Forest - ' + dataset_use + ' :')
        print(np.mean(rf_scores_all)) 
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, data_labels)
        # Initialize output confusion matrix
        rf_all_confmat = np.zeros((n_classes , n_classes))
        rf_all_kappa = []
        # Use split
        for train_index, test_index in skf.split(sat_data, data_labels):
           # Split into training and test set
           y_train, y_test = data_labels[train_index], data_labels[test_index]
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
        rf_confmat_all[dataset_use] = rf_all_confmat
        rf_mean_kappa[dataset_use] = np.mean(rf_all_kappa)
        print(rf_all_confmat)
        print('kappa = ', np.mean(rf_all_kappa))


# SAVE RESULTS
# kNN - cross validation
with open(os.path.join(dirname, 'data', knn_file), 'wb') as output:
    pickle.dump([knn_cv_all], output, pickle.HIGHEST_PROTOCOL)
# kNN - confusion matrices   
with open(os.path.join(dirname, 'data', knn_confmat_file), 'wb') as output:
    pickle.dump([knn_confmat_all], output, pickle.HIGHEST_PROTOCOL)
# RF
with open(os.path.join(dirname, 'data', rf_file), 'wb') as output:
    pickle.dump([rf_cv_all], output, pickle.HIGHEST_PROTOCOL)

# Save parameters
with open(os.path.join(dirname, 'data', classify_params_file), 'wb') as output:
    pickle.dump([knn_k, rf_ntrees], output, pickle.HIGHEST_PROTOCOL)


#%% Print number of instances for each class
n_class_samples = []
for key in class_dict.keys():
    val = class_dict[key]
    n_instances = length(data_labels[data_labels==val])
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

#%% Mean Accuracy - both in same plot (kNN and RF)
if combined_bar_plots:
    plt.figure()
    plt.bar(x_bars*2+ofs, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.bar(x_bars*2-ofs, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*n_datasets)
    plt.xticks(x_bars*2, list(rf_mean_acc.keys()))
    plt.yticks(np.linspace(0.1,1,num=10))
    plt.grid(True)
    plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k)+', normalization: '+norm_type)
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.legend(['Largest class %', 'RF', 'kNN'])
    plt.show()

#%% Mean Kappa - both in same plot (kNN and RF)
if plot_kappa:
    plt.figure()
    plt.bar(x_bars*2+ofs, list(rf_mean_kappa.values()), align='center', color='b', alpha=alf)
    plt.bar(x_bars*2-ofs, list(knn_mean_kappa.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars*2, list(rf_mean_kappa.keys()))
    plt.yticks(np.linspace(0.1,1,num=10))
    plt.grid(True)
    plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k)+', normalization: '+norm_type)
    plt.ylabel(r'Mean $\kappa$, n_splits = '+str(crossval_split_k)) 
    plt.legend(['RF', 'kNN'])
    plt.show()

# Mean Accuracy - Separate plots (kNN and RF)
if separate_bar_plots:
    # RF
    plt.figure()
    plt.bar(x_bars, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.title('RF, n_trees: '+str(rf_ntrees)+', normalization: '+norm_type)
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.show()
    # kNN
    plt.figure()
    plt.bar(x_bars, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars, list(knn_mean_acc.keys()))
    plt.yticks(np.linspace(0.1,1,num=10))
    plt.grid(True)
    plt.title('kNN, k: '+str(knn_k)+', normalization: '+norm_type)
    plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
    plt.show()
    
# Mean Accuracy - Separate plots for all datasets
if separate_dataset_plots:
    for dataset_id in id_list:
        key_list = []
        rf_accuracy = []
        knn_accuracy = []
        for dataset_key in list(rf_mean_acc.keys()):
            if dataset_key[-1] == dataset_id:
                key_list.append(dataset_key)
                rf_accuracy.append(rf_mean_acc[dataset_key])
                knn_accuracy.append(knn_mean_acc[dataset_key])
        
        if key_list:
            # Plot
            x_bars = np.arange(length(rf_accuracy))
            plt.figure()
            plt.bar(x_bars*2+ofs, rf_accuracy , align='center', color='b', alpha=alf)
            plt.bar(x_bars*2-ofs, knn_accuracy, align='center', color='r', alpha=alf)
            plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*length(rf_accuracy))
            plt.xticks(x_bars*2, key_list )
            plt.yticks(np.linspace(0.1,1,num=10))
            plt.grid(True)
            plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k)+', normalization: '+norm_type+
                      '\n Min:'+', live = '+str(min_p_live)+', defo = '+
                       str(min_p_defo)+', diff = '+str(diff_live_defo)+', trees = '+str(min_tree_live))
            plt.ylabel('Mean accuracy, n_splits = '+str(crossval_split_k)); plt.ylim((0,1))
            plt.legend(['Largest class %', 'RF', 'kNN'])
            plt.show()


#%% TEST ON COMPLETE IMAGE
#if plot_image_result:
#    ## Read satellite data
#    try:
#        # Read predefined file
#        with open(os.path.join(dirname, 'input-paths', sat_pathfile_name)) as infile:
#            sat_file = infile.readline().strip()
#            logit('Read file: ' + sat_file, log_type = 'default')
#        
#        # Load data
#        dataset = gdal.Open(sat_file)
#        gdalinfo_log(dataset, log_type='default')
#    except:
#        logit('Error, promt user for file.', log_type = 'default')
#        # Predefined file failed for some reason, promt user
#        root = tkinter.Tk() # GUI for file selection
#        root.withdraw()
#        sat_file = tkinter.filedialog.askopenfilename(title='Select input .tif file')
#        # Load data
#        dataset = gdal.Open(sat_file)
#        gdalinfo_log(dataset, log_type='default')
#    
#    # Read multiple bands
#    all_sat_bands = dataset.ReadAsArray()
#    
#    # Get bands used in two lists
#    bands_use_single = []
#    for key in input_data.modality_order: # .modality_bands.keys()
#        # Get list of lists of bands
#        bands_use_single += input_data.modality_bands[key]
#    
#    
#    ## Predict class for entire satellite image
#    sat_im = all_sat_bands[bands_use_single , :, : ]
#    ## Reshape array to n_cols*n_rows rows with the channels as columns 
#    sat_im_prediction, n_rows, n_cols = imtensor2array(sat_im)
#    
#    # NORMALIZE IMAGE - TODO: Change to 'local' both here and in get_sat_data??
#    sat_im_prediction = norm01(sat_im_prediction, norm_type=norm_type, log_type = 'print')
#    
#    # For colourbar: Get the list of unique class numbers  
#    class_n_unique = np.unique(list(class_dict.values()))
#    # Use the highest and lowest class n for colourbar visualization
#    class_n_lowest = np.min(class_n_unique) 
#    class_n_highest = np.max(class_n_unique)
#     
#    colors = ['red','green','blue','purple']
#    cmap = plt.get_cmap('jet', length(class_n_unique)) # Number of colours = n. of classes
#    
#                       
#    # kNN image
#    kNN_im_result = neigh.predict(sat_im_prediction)
#    # Reshape to original input size
#    sat_result_kNN = np.reshape(kNN_im_result, (n_rows, n_cols))
#    
#    # Show classification result
#    fig = plt.figure()
#    plt.imshow(sat_result_kNN.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
#    plt.colorbar(ticks=np.unique(list(class_dict.values())) )
#    plt.title('kNN result' + dataset_use)
#    plt.show()  # display it
#    
#    
#    # RF image
#    rf_im_result = rf_all.predict(sat_im_prediction)
#    # Reshape to original input size
#    sat_result_rf = np.reshape(rf_im_result, (n_rows, n_cols))
#    
#    fig2 = plt.figure()
#    plt.imshow(sat_result_rf.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
#    plt.colorbar(ticks=np.unique(list(class_dict.values())) )
#    plt.title('Random Forest result, '+dataset_use)
#    plt.show()  # display it
