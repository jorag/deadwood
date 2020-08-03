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


#%%
new_datalist_xls_file = '2020_proof_of_concept_datasets.xls' 

# List of datasets to process
dataset_list = ['refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3', 'PGNLM_20200225', 'PGNLM_20200227', 'geo_opt', 'NDVI'] # C3', 
#dataset_list = ['PGNLM_20200219', 'PGNLM_20200220', 'PGNLM_20200221', 'PGNLM_20200222','PGNLM_20200223', 'PGNLM_20200224', 'PGNLM_20200225', 'geo_opt']
id_list = ['A', 'C'] #['A', 'B', 'C'] # TODO: 20190909 Consider changing this a date string

# Datasets to add optical bands from
opt_dataset_list = ['geo_opt', 'NDVI']

# Which Sentinel-2 bands to use
#opt_bands_include = ['b02','b03','b04','b05','b06','b07','b08','b08a','b11','b12']
opt_bands_include = ['b02','b03','b04','b05','b08'] # all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

#%% Classification parameters
crossval_split_k = 5
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, shuffle=True, random_state=crossval_split_k)
knn_k = 5
rf_ntrees = 200 # Number of trees in the Random Forest algorithm

knn_mean_acc = dict()
rf_mean_acc = dict()

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
for row in df.itertuples(index=True, name='Pandas'):
    print(row.Path, row.gamma)

    dataset_use = row.Time+'-'+row.Dataset_ID 
        
    # Set name of output object
    # Use path from Excel file
    try: 
        sat_file = row.Path

        #print(sat_file)
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
        sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
        
        # Create a new array or append to existing one
        if i_defo == 0:
            defo_data = np.copy(sat_data)
        else:
            defo_data = np.hstack((defo_data, sat_data))
            
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
        sat_data = np.reshape(sat_data, (c_first_shape[0],c_first_shape[1]*c_first_shape[2]), order='C')
        
        # Create a new array or append to existing one
        if i_live == 0:
            live_data = np.copy(sat_data)
        else:
            live_data = np.hstack((live_data, sat_data))
    
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
    # Cross validate - kNN - All data
    knn_all = KNeighborsClassifier(n_neighbors=knn_k)
    knn_scores_all = cross_val_score(knn_all, x, y, cv=crossval_kfold)
    #print('kNN - ' + dataset_use + ' :')
    #print(np.mean(knn_scores_all))
    knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
    
    rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
    rf_scores_all = cross_val_score(rf_all, x, y, cv=crossval_kfold)
    print('Random Forest - ' + dataset_use + ' :')
    print(np.mean(rf_scores_all))
    rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        


#%% Plot classification summary
# Plot summary statistics
n_datasets = length(rf_mean_acc)
x_bars = np.arange(n_datasets) # range(n_datasets)
ofs = 0.25 # offset
alf = 0.7 # alpha

if False: #plot_rf_dataset_comp:
     # Disp
     id_list_use = ['A', 'C']
     c_vec2 = ['r', 'b', 'g']
     
     # Percent factor
     pct_f = 100
     
     #sar_names_dataset = ['IDAN_50_C3', 'boxcar_5x5_C3', 'refined_Lee_5x5_C3', 'PGNLM-20191224-1814', 'NDVI', 'optical']
     #sar_names_display = ['IDAN', 'boxcar', 'refined Lee', 'PGNLM', 'NDVI', 'optical']
     
     sar_names_dataset = ['IDAN_50_C3', 'boxcar_5x5_C3', 'refined_Lee_5x5_C3', 'PGNLM_20200227'] #
     sar_names_display = ['IDAN', 'boxcar', 'refined Lee', 'PGNLM'] # 'refined Lee'
     
     #sar_names_dataset = ['PGNLM_20200219', 'PGNLM_20200220', 'PGNLM_20200221', 'PGNLM_20200222','PGNLM_20200223', 'PGNLM_20200224', 'PGNLM_20200225']
     #sar_names_display = ['19', '20', '21', '22','23', '24', '25']
     
     opt_names_dataset = ['geo_opt', 'NDVI']
     opt_names_display = ['optical', 'NDVI'] # 'geo_opt'
     n_opt = length(opt_names_dataset)
     
     sar_data_dict = dict(zip(sar_names_dataset, sar_names_display)) 
    
     ofs_use = np.copy(ofs)
     plt.figure()
     x_bars = np.arange(length(sar_names_dataset)+ n_opt)
     #x_bars = np.arange(length(sar_names_dataset))
     #datakey_list = list(rf_mean_acc.keys())
     #datakey_list.sort()
     for i_dataset, dataset_id in enumerate(id_list_use):
        key_list = []
        rf_accuracy = []
        
        for dataset_key in sar_names_dataset:
            key_list.append(dataset_key+'-'+dataset_id)
            rf_accuracy.append(pct_f*rf_mean_acc[dataset_key+'-'+dataset_id])
            
        print(rf_accuracy)
        if key_list:
            ofs_use = ofs_use * -1
            # Plot
            plt.bar(x_bars[:-n_opt]*2+ofs_use, rf_accuracy , align='center', color=c_vec2[i_dataset], alpha=alf)
            #plt.bar(x_bars*2+ofs_use, rf_accuracy , align='center', color=c_vec[i_dataset], alpha=alf)
            #plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*length(rf_accuracy))
     
     key_list = []
     rf_accuracy = []
     for dataset_key in opt_names_dataset:
            key_list.append(dataset_key+'-A')
            rf_accuracy.append(pct_f*rf_mean_acc[dataset_key+'-'+dataset_id])
    
     print(rf_accuracy)
     plt.bar(x_bars[-n_opt:]*2, rf_accuracy , align='center', color='g', alpha=alf)
        
     # Get display names from dict
     xtick_list = sar_names_display + opt_names_display
     #for i_dataset, dataset_id in enumerate(id_list):
         
    
     plt.xticks(x_bars*2, xtick_list )
     plt.yticks(pct_f*np.linspace(0.1,1,num=10))
     plt.grid(True)
     #plt.title('RF, n_trees: '+str(rf_ntrees)+ ', normalization: '+norm_type+
     #         '\n Min:'+', live = '+str(min_p_live)+', defo = '+
     #          str(min_p_defo)+', trees = '+str(min_tree_live))
     plt.ylabel('Mean accuracy %, '+str(crossval_split_k)+'-fold cross validation'); plt.ylim((0,pct_f*1))
     plt.legend(['RS2 25 July 2017', 'RS2 1 August 2017', 'S2 26 July 2017'], loc='lower right')
    
     plt.show()
