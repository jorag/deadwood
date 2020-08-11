#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:49:32 2020

@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
import gdal
import pandas as pd
import os # Necessary for relative paths
import ast
import copy 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # train/test set split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # is deafault in cross-val?
from sklearn.metrics import confusion_matrix
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


#%% File with list of datasets and band info
# Path to PGNLM data
pgnlm_datadir = os.path.join(os.path.realpath('..'), 'gnlm34', 'data')
pgnlm_dataset_list = os.path.join(pgnlm_datadir, 'processed_list_v1.xls')

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

#%% Classification parameters
knn_k = 5
rf_ntrees = 200 # Number of trees in the Random Forest algorithm
# Type of split of training and test data
split_types = ['cross-val', 'AOI-largest-others'] # 'aoi_split'
# Max training test split imbalance fraction
split_max_imbalance = 0.6
# Cross validation parameters
crossval_split_k = 5
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, shuffle=True, random_state=crossval_split_k)

#%% Output (dicts)
params_dict = dict()

knn_mean_acc = dict()
rf_mean_acc = dict()

knn_aoi_all = dict()
knn_aoi_mean = dict()
knn_aoi_min = dict()
rf_aoi_all = dict()
rf_aoi_mean = dict()
rf_aoi_min = dict()

# %% Read defoliated AOI                          
# Get full data path from specified by input-path file
with open(os.path.join(dirname, 'input-paths', 'defo1-aoi-path')) as infile:
    defo_file = infile.readline().strip()
# Reading polygons into list
defo_aoi_list = read_wkt_csv(defo_file)

# %% Read live AOI                          
# Get full data path from specified by input-path file
with open(os.path.join(dirname, 'input-paths', 'live1-aoi-path')) as infile:
    live_file = infile.readline().strip()
# Reading polygons into list
live_aoi_list = read_wkt_csv(live_file)
                   
# %% Define band list (TODO: load from Excel file instead??)
lat_band = 9
lon_band = 10
sar_bands_use = [0, 1, 2, 3, 4]
# Features to use
c3_feature_type = 'c3pgnlm5feat'


datasets_xls = pd.ExcelFile(pgnlm_dataset_list)
df = pd.read_excel(datasets_xls)

# %% LOOP THROUGH SATELLITE DATA
for row in df.to_dict(orient="records"):
    try:
        # Set name of output object for use as key in dicts
        dataset_use = row['Time']+'-'+row['Dataset_ID']
        sat_file = row['Path']
    except:
        # Something went wrong
        print('Error!')
        continue
      
    # %% Try reading the dataset
    try:
        # Load data
        dataset = gdal.Open(sat_file)
        #gdalinfo_log(dataset, log_type='default')
        
        # Read ALL bands - note that it will be zero indexed
        raster_data_array = dataset.ReadAsArray()
    except:
        # Something went wrong
        continue
    
    # %% Store parameters in dict
    params_dict[dataset_use] = row
            
    # %% Get array with SAR data
    sat_data_temp = raster_data_array[sar_bands_use,:,:]   
        
            
    #%% Read lat and lon bands
    lat = raster_data_array[lat_band,:,:]
    lon = raster_data_array[lon_band,:,:]
    
    #%% Get defo AOI pixels
    for i_defo, defo_aoi in enumerate(defo_aoi_list):
        # Get LAT and LONG for bounding rectangle and get pixel coordinates 
        lat_bounds, lon_bounds = defo_aoi.bounding_coords()
        row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
        x_min = row_ind[0]; x_max = row_ind[1]
        y_min = col_ind[0]; y_max = col_ind[1]

        # Get SAR data from area
        sat_data = sat_data_temp[:, x_min:x_max, y_min:y_max]
        
        # Extract SAR covariance mat features, reshape to get channels last
        sat_data = np.transpose(sat_data, (1,2,0))
        sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
        
        # Flatten, reshape to channels first due to ordering of reshape
        sat_data = np.transpose(sat_data, (2,0,1))
        c_first_shape = sat_data.shape
        n_samples = c_first_shape[1]*c_first_shape[2]
        sat_data = np.reshape(sat_data, (c_first_shape[0], n_samples), order='C')
        
        # Create a new array or append to existing one
        if i_defo == 0:
            defo_data = np.copy(sat_data)
            defo_aoi_ind = np.zeros(n_samples)
            defo_aoi_samples = {i_defo : n_samples}
        else:
            defo_data = np.hstack((defo_data, sat_data))
            defo_aoi_ind = np.append(defo_aoi_ind, i_defo*np.ones((n_samples, 1)) )
            defo_aoi_samples[i_defo] = n_samples
            
    #%% Get live AOI pixels
    for i_live, live_aoi in enumerate(live_aoi_list):
        # Get LAT and LONG for bounding rectangle and get pixel coordinates 
        lat_bounds, lon_bounds = live_aoi.bounding_coords()
        row_ind, col_ind = geobox(lat_bounds, lon_bounds, lat, lon, margin=(0,0), log_type='default')
        x_min = row_ind[0]; x_max = row_ind[1]
        y_min = col_ind[0]; y_max = col_ind[1]

        # Get SAR data from area
        sat_data = sat_data_temp[:, x_min:x_max, y_min:y_max]
        
        # Extract SAR covariance mat features, reshape to get channels last
        sat_data = np.transpose(sat_data, (1,2,0))
        sat_data = get_sar_features(sat_data, feature_type=c3_feature_type, input_type='img')
        
        # Flatten, reshape to channels first due to ordering of reshape
        sat_data = np.transpose(sat_data, (2,0,1))
        c_first_shape = sat_data.shape
        n_samples = c_first_shape[1]*c_first_shape[2]
        sat_data = np.reshape(sat_data, (c_first_shape[0], n_samples), order='C')
        
        # But this loop has to be run several times if AOI split unless constant AOI split of training and test...
        if 'AOI-largest-others' in split_types:
            # Create a new array or append to existing one
            if i_live == 0:
                live_data = np.copy(sat_data)
                live_aoi_ind = np.zeros(n_samples)
                live_aoi_samples = {i_live : n_samples}
            else:
                live_data = np.hstack((live_data, sat_data))
                live_aoi_ind = np.append(live_aoi_ind, i_live*np.ones((n_samples, 1)) )
                live_aoi_samples[i_live] = n_samples
    
    #%% Plot 3D backscatter values

    # Merge arrays with live and defoliated data
    #x = np.hstack((live_data, defo_data))
    x = np.hstack((defo_data, live_data))
    x = x.transpose((1,0))
    # Create labels
    #y = np.hstack((1*np.ones(length(live_data),dtype='int'), 0*np.ones(length(defo_data),dtype='int')))
    y = np.hstack((0*np.ones(length(defo_data),dtype='int'), 1*np.ones(length(live_data),dtype='int')))
    
    labels_dict = None # dict((['live', 'defo'], ['live', 'defo']))
    
    # Plot
    #modalitypoints3d('reciprocity', x, y, labels_dict=labels_dict, title=dataset_use)
    
    #%% Classify
    if 'cross-val' in split_types:
        # Cross validate - kNN - All data
        knn_all = KNeighborsClassifier(n_neighbors=knn_k)
        knn_scores_all = cross_val_score(knn_all, x, y, cv=crossval_kfold)
        #print('kNN - ' + dataset_use + ' :')
        #print(np.mean(knn_scores_all))
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, x, y, cv=crossval_kfold)
        #print('Random Forest - ' + dataset_use + ' :')
        #print(np.mean(rf_scores_all))
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
    if 'AOI-largest-others' in split_types:
        # Create training and test datasets
        
        #%% LIVE
        # Find max number of samples in an AOI 
        live_samples_n = list(live_aoi_samples.values())
        max_live_samples = np.max(live_samples_n)
        # Split so that largest AOI is in one array and the rest in another
        live_others_aoi = []
        for key, value in live_aoi_samples.items():
            if value == max_live_samples:
                # Largest AOI
                live_largest_aoi = np.squeeze(live_data[:, np.where(live_aoi_ind == key)])
            else:
                # Add to other combination
                try:
                    live_others_aoi = np.hstack((live_others_aoi, np.squeeze(live_data[:, np.where(live_aoi_ind == key)]) ))
                except:
                    live_others_aoi = np.squeeze(live_data[:, np.where(live_aoi_ind == key)])
                    
        #%% DEFO
        # Find max number of samples in an AOI 
        defo_samples_n = list(defo_aoi_samples.values())
        max_defo_samples = np.max(defo_samples_n)
        # Split so that largest AOI is in one array and the rest in another
        defo_others_aoi = []
        for key, value in defo_aoi_samples.items():
            if value == max_defo_samples:
                # Largest AOI
                defo_largest_aoi = np.squeeze(defo_data[:, np.where(defo_aoi_ind == key)])
            else:
                # Add to other combination
                try:
                    defo_others_aoi = np.hstack((defo_others_aoi, np.squeeze(defo_data[:, np.where(defo_aoi_ind == key)]) ))
                except:
                    defo_others_aoi = np.squeeze(defo_data[:, np.where(defo_aoi_ind == key)])
                    print(defo_others_aoi.shape)
        
        #%% Classify
        # Create combinations
        x_oo = np.hstack((defo_others_aoi, live_others_aoi)).transpose((1,0))
        y_oo = np.hstack((0*np.ones(length(defo_others_aoi),dtype='int'), 1*np.ones(length(live_others_aoi),dtype='int')))
        x_lo = np.hstack((defo_largest_aoi, live_others_aoi)).transpose((1,0))
        y_lo = np.hstack((0*np.ones(length(defo_largest_aoi),dtype='int'), 1*np.ones(length(live_others_aoi),dtype='int')))
        x_ol = np.hstack((defo_others_aoi, live_largest_aoi)).transpose((1,0))
        y_ol = np.hstack((0*np.ones(length(defo_others_aoi),dtype='int'), 1*np.ones(length(live_largest_aoi),dtype='int')))
        x_ll = np.hstack((defo_largest_aoi, live_largest_aoi)).transpose((1,0))
        y_ll = np.hstack((0*np.ones(length(defo_largest_aoi),dtype='int'), 1*np.ones(length(live_largest_aoi),dtype='int')))
        
        # kNN classifier
        print('------ kNN - ' + dataset_use + ' --------')
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        neigh.fit(x_oo, y_oo) # Fit kNN
        a_oo_ll = neigh.score(x_ll, y_ll)
        print('Accuracy, OO-LL: '+' = ', a_oo_ll) # Score kNN
        #prediction_result = neigh.predict(x_ll) # Test kNN on test dataset
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        neigh.fit(x_ol, y_ol) # Fit kNN
        a_ol_lo = neigh.score(x_lo, y_lo)
        print('Accuracy, OL-LO: '+' = ', a_ol_lo) # Score kNN
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        neigh.fit(x_lo, y_lo) # Fit kNN
        a_lo_ol = neigh.score(x_ol, y_ol)
        print('Accuracy, LO-OL: '+' = ', a_lo_ol) # Score kNN
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        neigh.fit(x_ll, y_ll) # Fit kNN
        a_ll_oo = neigh.score(x_oo, y_oo)
        print('Accuracy, LL-OO: '+' = ', a_ll_oo) # Score kNN
        print('Accuracy, cross: '+' = ', np.mean(knn_scores_all))
        
        # Add mean to output dict
        knn_aoi_all[dataset_use] = [a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo]
        knn_aoi_mean[dataset_use] = np.mean([a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo])
        knn_aoi_min[dataset_use] = np.min([a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo])
        
        
        # RF classifier
        rf = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf.fit(x_oo, y_oo) # Fit kNN
        a_oo_ll = rf.score(x_ll, y_ll)
        #prediction_result_rf = rf.predict(x_ll) # Test RF on test dataset
        rf = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf.fit(x_ol, y_ol) 
        a_ol_lo = rf.score(x_lo, y_lo)
        rf = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf.fit(x_lo, y_lo) 
        a_lo_ol = rf.score(x_ol, y_ol)
        rf = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf.fit(x_ll, y_ll) 
        a_ll_oo = rf.score(x_oo, y_oo)
        
        # Add mean to output dict
        rf_aoi_all[dataset_use] = [a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo]
        rf_aoi_mean[dataset_use] = np.mean([a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo])
        rf_aoi_min[dataset_use] = np.min([a_oo_ll, a_ol_lo, a_lo_ol, a_ll_oo])


#%% Plot classification summary
# Plot summary statistics
n_datasets = length(rf_mean_acc)

plt.figure()
plt.plot(list(rf_mean_acc.values()), 'b')
plt.plot(list(knn_mean_acc.values()), 'r')
plt.title('RF and kNN - cross-val')

plt.figure()
plt.plot(list(knn_aoi_mean.values()), 'b')
plt.plot(list(knn_mean_acc.values()), 'r')
plt.plot(list(knn_aoi_min.values()), 'g')
plt.title('kNN - AOI and cross-val')

plt.figure()
plt.plot(list(rf_aoi_mean.values()), 'b')
plt.plot(list(rf_mean_acc.values()), 'r')
plt.plot(list(rf_aoi_min.values()), 'g')
plt.title('RF - AOI and cross-val')

#%% Plot for different parameters

# Key to use for sorting (lines/points in plot)
sort_key = 'n_small'
x_axis = 'opt_percentile'
x_lim = (0,100) # Limit for a_axis
acc_measure = rf_aoi_min


# Two dicts, but use same key (numbers) - one dict for params and one for average values
# Also a list of dicts, for finding index? - No, use list(dict.values)
# 
params_plot = dict() # replace with a list
values_plot = dict()
paramset_counter = 0 
# Output a list of values_plot dicts??
for dataset_use, params in params_dict.items():
    # Delete dataset key to enable averaging over datasets with the same params
    curr_dict = copy.deepcopy(params)
    # Remove identifying fields
    curr_dict.pop('Dataset_ID')
    curr_dict.pop('Path')
    curr_dict.pop('Time')
    curr_dict.pop('opt_thresh')
    curr_dict.pop('thresh')
    # for p_comp in params_compare_list:
        # if params[p_comp] ...
    
    # Check if parameter set has been used before    
    if curr_dict in list(params_plot.values()):
        # Find matching dict
        match_ind = list(params_plot.values()).index(curr_dict)
        match_ind = str(match_ind)
        print(match_ind)
        # Get corresponding key (to allow keys other than integers matching list index)
        # - or use assume integer indice and use match_ind diretly? -Easiest, start with this
        prev_acc = values_plot[match_ind]['acc']
        n_sets = values_plot[match_ind]['n_sets']
        values_plot[match_ind]['acc'] = n_sets*prev_acc/(n_sets+1) + acc_measure[dataset_use]/(n_sets+1)
        # Update number of sets used 
        values_plot[match_ind]['n_sets'] = n_sets+1
    else:
        # Key
        key_use = str(paramset_counter)
        # New parameter set, populate it
        values_plot[key_use] = dict()
        values_plot[key_use]['acc'] = acc_measure[dataset_use]
        values_plot[key_use]['x_axis'] = params[x_axis]
        values_plot[key_use]['n_sets'] = 1
        # Add to dict of parameter set
        params_plot[key_use] = curr_dict
        # Update counter used as keys
        paramset_counter += 1 
        
    
# Plot output lists
# TODO: Map x_axis values to colour and marker
cvec, mvec = mycolourvec(markers=True)
#c_iter = 0
plt.figure()
for value in values_plot.values():
    # value = res_dict # loop over for res_dict in avg_list:
    plt.plot(value['x_axis'], value['acc'], 'kx')
    #c_iter += 1
plt.xlim(x_lim)
    
#%% Plot for different parameters

# Key to use for sorting (lines/points in plot)
sort_key = 'n_small'
x_axis = 'opt_percentile'
x_lim = (0,100) # Limit for a_axis
acc_measure = rf_aoi_min

# Initialize output lists
plot_dict = dict()
for dataset_use, params in params_dict.items():
    plot_dict[params[sort_key]] = {'acc': [] , 'x_axis' : []}

    
# Populate output lists
prev_dict = dict()
#avg_counter = -1 # Ensure results are accumulated from first iter
avg_counter = dict() # count number of datasets for a given parameter
temp_acc = []
avg_list = [] # or dict instead since adding an average value to dict will ruin comparison..?
# Two dicts, but use same key (numbers) - one dict for params and one for average values
# Also a list of dicts, for finding index? - No, use list(dict.values)
params_plot = dict()
values_plot = dict()
for dataset_use, params in params_dict.items():
    # Delete dataset key to enable averaging over datasets with the same params
    curr_dict = copy.deepcopy(params)
    # Remove identifying fields
    curr_dict.pop('Dataset_ID')
    curr_dict.pop('Path')
    curr_dict.pop('Time')
    curr_dict.pop('opt_thresh')
    curr_dict.pop('thresh')
    temp_acc.append(acc_measure[dataset_use])
    # for p_comp in params_compare_list:
        # if params[p_comp] ...
    
    if curr_dict == prev_dict or avg_counter == -1:
        # Average elements
        avg_counter += 1
        print(avg_counter)
    else:
        print(prev_dict)
        print(curr_dict)
        # Populate elements
        plot_dict[params[sort_key]]['acc'].append(np.mean(temp_acc))
        plot_dict[params[sort_key]]['x_axis'].append(params[x_axis])
        avg_counter = 0
        temp_acc = []
    # For comparison and averaging
    prev_dict = copy.deepcopy(curr_dict)

# Ensure that final datasets are included
plot_dict[params[sort_key]]['acc'].append(np.mean(temp_acc))
plot_dict[params[sort_key]]['x_axis'].append(params[x_axis])
    
# Plot output lists
cvec, mvec = mycolourvec(markers=True)
c_iter = 0
plt.figure()
for key, value in plot_dict.items():
    print(key)
    # value = res_dict # loop over for res_dict in avg_list:
    print(length(value['acc']))
    plt.plot(value['x_axis'], value['acc'], cvec[c_iter]+mvec[c_iter])
    c_iter += 1
plt.xlim(x_lim)
