#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:49:32 2020

@author: jorag
"""

import matplotlib # For setting font size
import matplotlib.pyplot as plt
import numpy as np
import gdal
import pandas as pd
import os # Necessary for relative paths
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, 
                                     StratifiedKFold,
                                     GroupKFold)
from sklearn.metrics import confusion_matrix
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


#%% File with list of datasets and band info  
new_datalist_xls_file = 'SF_forest_subsets.xls' 
# Prefix for output datamodalities object filename
datamod_fprefix = 'PGNLM-SNAP_C3_20200116' #'PGNLM-SNAP_C3_geo_OPT_20200113'
base_obj_name = 'DiffGPS_FIELD_DATA'+'.pkl' # Name of the (pure) field data object everything is based on

# Run on grid data or homogeneous AOI
use_test_aois = True

pgnlm_set = 'PGNLM_19-2_v4'
# List of datasets to process
dataset_list = ['refined_Lee_5x5_C3', 'boxcar_5x5_C3', 'IDAN_50_C3', 'NLSAR_1_1', pgnlm_set, 'PGNLM_19-2_v256'] # 'C3', 'NDVI'
#dataset_list = [pgnlm_set, 'boxcar_5x5_C3', 'refined_Lee_5x5_C3', 'IDAN_50_C3', 'NLSAR_1_1', pgnlm_set]
#dataset_keys = ['optical', 'boxcar',  'refined Lee', 'IDAN', 'NL-SAR', 'PGNLM']
id_list = ['A', 'C'] 

# Datasets to add optical bands from
opt_dataset_list = ['PGNLM_19-2_v256']

# Which Sentinel-2 bands to use
opt_bands_include = ['b02','b03','b04','b08'] # all 10 m resolution
    
# Path to working directory 
dirname = os.path.realpath('.') 
parent_dir = os.path.realpath('..') # For parent directory 

#%% Classification parameters
knn_k = 5
rf_ntrees = 200 # Number of trees in the Random Forest algorithm
# Cross validation parameters
crossval_split_k = 4
crossval_kfold = StratifiedKFold(n_splits=crossval_split_k, 
                                 shuffle=True, random_state=crossval_split_k)
group_kfold = GroupKFold(n_splits=crossval_split_k)

#%% Plotting and figure options

# Version ID, changes to these options should change version number 
# changes should also be commited to Github to store exact settings
version_id = 'v3' # LAST UPDATE 20201117 - only use 10m optical bands, use coregistered optical image

# Figure options
plt_fontsize = 12
plt_height = 6.0 # inches
plt_width = 7.8 # inches
plt_dpi = 200 # Dots Per Inch (DPI)


#%% Output dicts
knn_mean_acc = dict()
knn_all_acc = dict()
rf_mean_acc = dict()
rf_all_acc = dict()

data_dict = dict()


# %% AOIs or grid data?
if use_test_aois:                        
    # Get full data path from specified by input-path files
    with open(os.path.join(dirname, 'input-paths', 'defo1-aoi-path')) as infile:
        defo_file = infile.readline().strip()
    with open(os.path.join(dirname, 'input-paths', 'live1-aoi-path')) as infile:
        live_file = infile.readline().strip()
    # Read AOI polygons into lists
    data_dict['dead'] = read_wkt_csv(defo_file)  
    data_dict['live'] = read_wkt_csv(live_file)
    # Set cross validator to use and groups param
    crossval_use = crossval_kfold
    groups = None
else:
    # Get full data path from specified by input-path files
    with open(os.path.join(dirname, 'input-paths', 'mixed_30m_latlon')) as infile:
        grid_file = infile.readline().strip()
    with open(os.path.join(dirname, 'input-paths', 'mixed_30m_classes')) as infile:
        class_file = infile.readline().strip()
    # Read layer
    grid_list = read_shp_layer(grid_file)
    # Read list of class IDs
    class_dict = read_class_csv(class_file)
    
    # Create lists for each class
    #states_use = ['live', 'dead', 'damaged', 'other']
    #states_use = ['live', 'dead', 'other']
    #states_use = ['live', 'dead', 'damaged']
    states_use = ['live', 'dead']
    for state_collect in states_use:
        data_dict[state_collect] = layer2roipoly_state(grid_list, state_collect)
        
    # Set cross validator to use and groups param
    crossval_use = group_kfold
    groups = []
    
                   
# %% Load band lists from Excel file
xls_fullpath = os.path.join(dirname, 'input-paths', new_datalist_xls_file)
datasets_xls = pd.ExcelFile(xls_fullpath)
df = pd.read_excel(datasets_xls)


# %% LOOP THROUGH SATELLITE DATA
for dataset_id in id_list:
    for dataset_in in dataset_list:
        dataset_use = dataset_in +'-'+dataset_id 
            
        # Set name of output object
        # Use path from Excel file
        try: 
            sat_file = df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Path'].values[0]
            # Get indices for bands, lat, lon, SAR, and optical
            lat_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lat_band'].values[0])[0]
            lon_band = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) & (df['Processing_key'] == dataset_in), 'Lon_band'].values[0])[0]
            sar_bands_use = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) 
                & (df['Processing_key'] == dataset_in), 'SAR_bands'].values[0])
            opt_bands_use = ast.literal_eval(df.loc[(df['Dataset_key'] == dataset_id) 
                & (df['Processing_key'] == dataset_in), 'OPT_bands'].values[0])

            # If optical bands part of dataset, assume all are present and create dict
            # TODO 20200113 - Fix this assumption
            if opt_bands_use:
                # List of optical band names (added zero for correct alphabetical sorting)
                opt_band_names = ['b02','b03','b04','b08']
                
                # Add OPT bands
                opt_bands_dict = dict()
                opt_bands_dict[dataset_use] = dict(zip(opt_band_names , opt_bands_use))
                
            #print(sat_file)
        except:
            # Something went wrong
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
        
        # %% Check input data type for feature extraction
        if dataset_in.lower()[0:5] in ['pgnlm']:
            if dataset_in.lower()[6:10] in ['2019', 'best']:
                c3_feature_type = 'iq2c3'
            else:
                c3_feature_type = 'c3pgnlm5feat'
        elif dataset_in.lower()[0:5] in ['nlsar']:
            c3_feature_type = 'all' # 5 features
        elif dataset_in.lower()[-2:] in ['c3']:
            c3_feature_type = 'c3snap5feat'
        else:
            c3_feature_type = 'all'
                
        # %% If SAR data should be added
        if sar_bands_use:
            # Get array with SAR data
            sat_data_temp = raster_data_array[sar_bands_use,:,:]   
            
        # %% If MULTISPECTRAL OPTICAL data should be added
        if dataset_in in opt_dataset_list:
            opt_bands_use = [] # Check which of the available bands should be included 
            for key in opt_bands_include:
                opt_bands_use.append(opt_bands_dict[dataset_use][key])
            sat_data_temp = raster_data_array[opt_bands_use,:,:]
            
            # Calculate NDVI
            if dataset_in.lower()[0:4] in ['ndvi']:
                print('sat_data_temp.shape = ', sat_data_temp.shape)
                nir_pixels = sat_data_temp[opt_bands_include.index('b08'), :, :] 
                red_pixels = sat_data_temp[opt_bands_include.index('b04'), :, :]
                sat_data_temp = (nir_pixels-red_pixels)/(nir_pixels+red_pixels)
                sat_data_temp = sat_data_temp[np.newaxis, :, :]
                print('sat_data_temp.shape = ', sat_data_temp.shape)
                
        #%% Read lat and lon bands
        lat = raster_data_array[lat_band,:,:]
        lon = raster_data_array[lon_band,:,:]
        
        #%% Get AOI pixels

        # Initialise 
        x = None
        y = None
        #groups = np.array(())
        i_class_label = int(0)
        i_group = 0
        
        # Go through all classes
        for aoi_state, aoi_list in data_dict.items():
            # Loop through all AOIs for each class
            for i_aoi, aoi in enumerate(aoi_list):
                # Get LAT and LONG for bounding rectangle and get pixel coordinates 
                lat_bounds, lon_bounds = aoi.bounding_coords()
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
                
                #groups.append(i_group*np.ones(sat_data.shape[1],))
                
                # Create a new array or append to existing one
                if i_aoi == 0:
                    state_data = np.copy(sat_data)
                    # Check if groups array should initialized
                    if i_group == 0:
                        groups = i_group*np.ones(sat_data.shape[1],)
                    else:
                        groups = np.hstack((groups, i_group*np.ones(sat_data.shape[1],)))
                else:
                    state_data = np.hstack((state_data, sat_data))
                    groups = np.hstack((groups, i_group*np.ones(sat_data.shape[1],)))
                    
                i_group += 1
            
            # Add to merged data array
            if i_class_label == 0:
                x = np.copy(state_data)
                # TODO: 20201029 change to use x.shape instead of length
                y = i_class_label*np.ones( (length(state_data), ) ,dtype='int')
            else:
                x = np.hstack((x, state_data))
                y = np.hstack((y, i_class_label*np.ones( (length(state_data), ) ,dtype='int')))
            # Update class label
            i_class_label += int(1)
            
            
        # Transpose
        x = x.transpose((1,0))

        #%% Plot 3D backscatter values

        # Plot
        #labels_dict = None # dict((['live', 'defo'], ['live', 'defo']))
        #modalitypoints3d('reciprocity', x, y, labels_dict=labels_dict, title=dataset_use)
        
        #%% Classify
        group_kfold.get_n_splits(X=x, y=y, groups=groups)
        # Cross validate - kNN - All data
        knn_all = KNeighborsClassifier(n_neighbors=knn_k)
        knn_scores_all = cross_val_score(knn_all, x, y, groups=groups, 
                                         cv=crossval_use)
        #knn_scores_all = cross_val_score(knn_all, x, y, cv=crossval_kfold)
        print('kNN - ' + dataset_use + ' :')
        print(np.mean(knn_scores_all))
        knn_mean_acc[dataset_use] = np.mean(knn_scores_all)
        knn_all_acc[dataset_use] = knn_scores_all
        
        rf_all = RandomForestClassifier(n_estimators=rf_ntrees, random_state=0)
        rf_scores_all = cross_val_score(rf_all, x, y, groups=groups, 
                                        cv=crossval_use)
        #rf_scores_all = cross_val_score(rf_all, x, y, cv=crossval_kfold)
        print('Random Forest - ' + dataset_use + ' :')
        print(np.mean(rf_scores_all))
        rf_mean_acc[dataset_use] = np.mean(rf_scores_all)
        rf_all_acc[dataset_use] = rf_scores_all
        
#%% Plot classification summary
# Plot summary statistics
n_datasets = length(rf_mean_acc)
x_bars = np.arange(n_datasets) # range(n_datasets)
ofs = 0.25 # offset
alf = 0.7 # alpha

class_n, class_counts = np.unique(y, return_counts = True)


 # Disp
id_list_use = ['A', 'C']
c_vec2 = ['r', 'b', 'g']

# Percent factor
pct_f = 100

#sar_names_dataset = ['IDAN_50_C3', 'boxcar_5x5_C3', 'refined_Lee_5x5_C3', 'PGNLM-20191224-1814', 'NDVI', 'optical']
#sar_names_display = ['IDAN', 'boxcar', 'refined Lee', 'PGNLM', 'NDVI', 'optical']

sar_names_dataset = ['IDAN_50_C3', 'boxcar_5x5_C3', 'refined_Lee_5x5_C3', 'NLSAR_1_1', 'PGNLM_19-2_v4'] #
sar_names_display = ['IDAN', 'boxcar', 'refined Lee', 'NL-SAR', 'PGNLM'] # 'refined Lee'

opt_names_dataset = ['PGNLM_19-2_v256'] # opt_names_dataset = ['geo_opt', 'NDVI']
opt_names_display = ['optical'] # opt_names_display = ['optical', 'NDVI']
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
   yerr = [[],[]]
   
   for dataset_key in sar_names_dataset:
       key_list.append(dataset_key+'-'+dataset_id)
       rf_accuracy.append(pct_f*rf_mean_acc[dataset_key+'-'+dataset_id])
       yerr[0].append(pct_f *(min(rf_all_acc[dataset_key+'-'+dataset_id])- 
           rf_mean_acc[dataset_key+'-'+dataset_id]))
       yerr[1].append(pct_f* (max(rf_all_acc[dataset_key+'-'+dataset_id])-
           rf_mean_acc[dataset_key+'-'+dataset_id]))
       
   print(rf_accuracy)
   print(yerr)
   yerr = np.abs(yerr)
   if key_list:
       ofs_use = ofs_use * -1
       # Plot
       plt.bar(x_bars[:-n_opt]*2+ofs_use, rf_accuracy, yerr=yerr, 
               align='center', color=c_vec2[i_dataset], alpha=alf, 
               error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2))
       #plt.bar(x_bars*2+ofs_use, rf_accuracy , align='center', color=c_vec[i_dataset], alpha=alf)
       #plt.hlines(np.max(n_class_samples)/np.sum(n_class_samples), -1, 2*length(rf_accuracy))

key_list = []
rf_accuracy = []
yerr = [[],[]]
for dataset_key in opt_names_dataset:
       key_list.append(dataset_key+'-A')
       rf_accuracy.append(pct_f*rf_mean_acc[dataset_key+'-'+dataset_id])
       yerr[0].append(pct_f *(min(rf_all_acc[dataset_key+'-'+dataset_id])- 
           rf_mean_acc[dataset_key+'-'+dataset_id]))
       yerr[1].append(pct_f* (max(rf_all_acc[dataset_key+'-'+dataset_id])-
           rf_mean_acc[dataset_key+'-'+dataset_id]))

print(rf_accuracy)
print(yerr)
yerr = np.abs(yerr)
plt.bar(x_bars[-n_opt:]*2, rf_accuracy, yerr=yerr, align='center', 
        color='g', alpha=alf, 
        error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2))
    
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
 

#%% Save figure

# Set font size
matplotlib.rc('font', size=plt_fontsize)
# Set output directory
fig_dir = os.path.join(parent_dir, 'tempresults')
# Create filename
if use_test_aois:
    fname_out = 'AOI-accbar_'+pgnlm_set+'_RF-'+str(rf_ntrees)+'_k'+str(crossval_split_k)
else:
    fname_out = 'GRID-accbar_'+pgnlm_set+'_RF-'+str(rf_ntrees)+'_k'+str(crossval_split_k)

# Add classes to filename
for cname in data_dict.keys():
    fname_out += '_' + cname
# Add version and file type
fname_out += '_' + version_id+'.png'
   
# now, before saving to file:
figure = plt.gcf() # get current figure
figure.set_size_inches(plt_width, plt_height) 
# when saving, specify the DPI
plt.savefig(os.path.join(fig_dir, fname_out), dpi = plt_dpi) 
