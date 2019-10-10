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
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# PROCESSING PARAMETERS
crossval_split_k = 3
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k)
knn_k = 3
rf_ntrees = 15 # Number of trees in the Random Forest algorithm
separate_bar_plots = False # Combination of RF and kNN result, or separate plots

# Image plot parameter - TODO: Differentiate for SAR and optical data
norm_type = 'local' # 'global' # 

# Prefix for output cross validation object filename
crossval_fprefix = 'new-kNN' + str(knn_k) + 'trees' + str(rf_ntrees)

# Prefix for object filename
datamod_fprefix = 'All-data-0919'
id_list = ['A', 'B', 'C']

# Name of input object and file with satellite data path string
obj_in_name = datamod_fprefix + '-' + '.pkl'

## Read DataModalities object with ground in situ vegetation data
with open(os.path.join(dirname, 'data', obj_in_name), 'rb') as input:
    all_data = pickle.load(input)

# Set class labels for dictionary - TODO: Consider moving this to get_stat_data
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])
#class_dict_in = None

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
data_labels[np.where(y_data[:,1]>y_data[:,0])] = 2 # Defoliated
#data_labels[np.where(y_data[:,1]>0.05)] = 2 # Defoliated
data_labels[np.where(y_data[:,1]<=0.075)] = 0 # Other
data_labels[np.where(y_data[:,0] > 0.075)] = 1 # Live
data_labels[np.where(y_data[:,2]<=2)] = 0 # Other
#data_labels = np.random.randint(3, size=(length(y_data)))
class_dict=class_dict_in
n_classes = length(class_dict)

# Plot classifier result for entire image
plot_image_result = False

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


# Show PLC and PDC with class colours
# TODO: 3D plot with n_trees as one axis
plot_labels = labels.astype(int)
# Get standard colour/plotstyle vector
c_vec = mycolourvec()
# Convert to numpy array for indexation
c_vec = np.asarray(c_vec)
# Plot x
xs = np.arange(length(y_data)) # range(n_datasets)
fig = plt.figure()
plt.scatter(y_data[:,0], y_data[:,1], c=c_vec[plot_labels], marker='o', label=plot_labels)
plt.xlabel('Live Crown Proportion'); plt.ylabel('Defoliated Crown Proportion')
plt.ylim((-0.1,1)); plt.xlim((-0.1,1))
#plt.legend(['other', 'Live', 'Defoliated'])
#plt.legend()
plt.show()

# TRAIN AND CROSS-VALIDATE
# Go through all satellite images and all data modalities in object
for dataset_id in id_list: 
    for dataset_type in all_data.all_modalities:
        print(dataset_type)            
        # Get satellite data
        try:
            dataset_use = dataset_type+'-'+dataset_id
            sat_data = all_data.read_data_points(dataset_use, modality_type=dataset_type)
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
        #print(sat_data)
        
        # Name of input object and file with satellite data path string
        sat_pathfile_name = dataset_use + '-path'
        
#        # Get labels and class_dict (in case None is input, one is created)
#        labels_out, class_dict = input_data.assign_labels(class_dict=class_dict_in)
#        n_classes = length(class_dict)
#        
#        # Get all data
#        sat_data, data_labels = input_data.read_data_array(['quad_pol', 'optical'], 'all') 
#        # Get SAR data
#        sar_data, data_labels = input_data.read_data_array(['quad_pol'], 'all') 
#        # Get OPT data
#        opt_data, data_labels = input_data.read_data_array(['optical'], 'all') 
        
        # Plot in 3D
        #modalitypoints3d('van_zyl', sat_data, labels, labels_dict=class_dict)
        
        #breakpoint = dummy 

        # Normalize data - should probably be done when data is stored in object...
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
        print('kNN OPT+SAR - ' + dataset_use + ' :')
        print(np.mean(knn_scores_all)) 
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, labels)
        # Initialize output confusion matrix
        knn_all_confmat = np.zeros((n_classes , n_classes ))
        # Use split
        for train_index, test_index in skf.split(sat_data, labels):
           # Split into training and test set
           y_train, y_test = labels[train_index], labels[test_index]
           X_train, X_test = sat_data[train_index], sat_data[test_index]
           # Fit classifier
           knn_all.fit(X_train, y_train)
           # Calculate confusion matrix
           conf_mat_temp = confusion_matrix(y_test, knn_all.predict(X_test))
           # Add contribution to overall confusion matrix
           knn_all_confmat += conf_mat_temp
           
        # Add to output dict
        knn_confmat_all[dataset_use] = knn_all_confmat
        print(knn_all_confmat)
        
        
        # Cross validate - Random Forest - All data
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, sat_data, data_labels, cv=crossval_kfold)
        # Add to output dict
        rf_cv_all[dataset_use] = rf_scores_all
        print('RF OPT+SAR - ' + dataset_use + ' :')
        print(np.mean(rf_scores_all)) 
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        
        # Get split for cofusion matrix calculation
        skf = StratifiedKFold(n_splits=crossval_split_k)
        skf.get_n_splits(sat_data, labels)
        # Initialize output confusion matrix
        rf_all_confmat = np.zeros((n_classes , n_classes ))
        # Use split
        for train_index, test_index in skf.split(sat_data, labels):
           # Split into training and test set
           y_train, y_test = labels[train_index], labels[test_index]
           X_train, X_test = sat_data[train_index], sat_data[test_index]
           # Fit classifier
           rf_all.fit(X_train, y_train)
           # Calculate confusion matrix
           conf_mat_temp = confusion_matrix(y_test, rf_all.predict(X_test))
           # Add contribution to overall confusion matrix
           rf_all_confmat += conf_mat_temp
           
        # Add to output dict
        rf_confmat_all[dataset_use] = rf_all_confmat
        print(rf_all_confmat)


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
# Linreg
# # Try sorting dictionaries alphabetically
# From: 
#sorted(rf_mean_acc, key=rf_mean_acc.get, reverse=True)
#sorted(linreg_pdc_r2, key=linreg_pdc_r2.get, reverse=True)

# Both
plt.figure()
plt.bar(x_bars*2+ofs, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
plt.bar(x_bars*2-ofs, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*n_datasets)
plt.xticks(x_bars*2, list(rf_mean_acc.keys()))
plt.title('RF, n_trees: '+str(rf_ntrees)+ ' - kNN, k: '+str(knn_k))
plt.ylim((0,1))
plt.legend(['Largest class %', 'RF', 'kNN'])
plt.show()

# Separate plots
if separate_bar_plots:
    # RF
    plt.figure()
    plt.bar(x_bars, list(rf_mean_acc.values()), align='center', color='b', alpha=alf)
    plt.xticks(x_bars, list(rf_mean_acc.keys()))
    plt.title('RF, n_trees: '+str(rf_ntrees))
    plt.ylim((0,1))
    plt.show()
    # kNN
    plt.figure()
    plt.bar(x_bars, list(knn_mean_acc.values()), align='center', color='r', alpha=alf)
    plt.xticks(x_bars, list(knn_mean_acc.keys()))
    plt.title('kNN, k: '+str(knn_k))
    plt.ylim((0,1))
    plt.show()



# TEST ON COMPLETE IMAGE
if plot_image_result:
    ## Read satellite data
    try:
        # Read predefined file
        with open(os.path.join(dirname, 'input-paths', sat_pathfile_name)) as infile:
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


