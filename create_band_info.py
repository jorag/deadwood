#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:14:43 2019

@author: jorag
"""
import matplotlib.pyplot as plt
import numpy as np
import gdal
import tkinter
from tkinter import filedialog
import tkinter.messagebox as tkmb
import pandas as pd
import os # Necessary for relative paths
import ast
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Dataset index
datasets_idx = -1

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Open Spreadsheat
xls_fullpath = os.path.join(dirname, 'input-paths', '2020_dataset_plot_overview.xls')
datasets_xls = pd.ExcelFile(xls_fullpath)
datasets_df = pd.read_excel(datasets_xls)
paths_in = list(datasets_df['Path']) 

# Read LAST line - OR index specific dataset
latest_dataset_path = paths_in[datasets_idx] 

# Read dataset in path found from 3rd column using GDAL
dataset = gdal.Open(latest_dataset_path)
gdalinfo_log(dataset, log_type='default')

# Read ALL bands - note that it will be zero indexed
# 20191127: IF-test here to switch between reading data from file, 
# or use output (possibly from memory) from filtering
raster_data_array = dataset.ReadAsArray()

# Loop through all dataset bands - like visandtest.showallbands(raster_data_array)
# - plot with dataset name, min and max values, and band number as title
# - plot subplot with previous, current and next band??
# - ask user if he wants to add band to a list (lat, lon, SAR, or opt), multiple choice

             
# Initialize output lists
sorting_dict = {'SAR_bands': [], 'OPT_bands': [], 'Lat_band': [], 
'Lon_band': [] , 'unsure': []} #

# List for question dialogue
option_list = list(sorting_dict.keys())

# Rearrage dimensions to x,y,channel format
im_generate = np.transpose(raster_data_array, (1,2,0))
selection = 0

for i_band in range(im_generate.shape[2]):
    # Get statistics
    band_min = np.nanmin(im_generate[:,:,i_band])
    band_mean = np.nanmean(im_generate[:,:,i_band])
    band_max = np.nanmax(im_generate[:,:,i_band])
    # Normalization
    im_generate[:,:,i_band] =  norm01(im_generate[:,:,i_band], min_cap=-990,  min_cap_value=np.NaN)
    # Band title text
    band_txt = 'BAND '+ str(i_band) 
    # Plot data
    plt.figure(i_band)
    plt.imshow(im_generate[:,:,i_band], cmap='gray')
    plt.title(band_txt +' Min = '+str(band_min)+' Mean = '+str(band_mean)+' Max = '+str(band_max))
    plt.pause(0.05) # Make sure plot is displayed
    
    # Ask which band it its
    which_list, selection = ask_multiple_choice_question('Which band type is this?', option_list, title=band_txt, default_v = selection)
    
    # Store result
    sorting_dict[which_list].append(i_band)
    
    plt.close(i_band)
    
# Promt user and ask if band list should be stored
# - if there are already band lists, show the current list and the pre-existing one

# Add band lists to data frame
for band_type in option_list:
    try:
        #current_var = datasets_df.loc[datasets_df.index[datasets_idx], band_type]
        #print(ast.literal_eval(current_var))
        # Convert to object type to enable list input
        datasets_df[band_type] = datasets_df[band_type].astype('object')
        # Due to warning
        datasets_df.loc[datasets_df.index[datasets_idx], band_type] = sorting_dict[band_type]
    except:
         print(band_type + ' not in file!') 


# Add band lists to data frame
for band_type in option_list:
    try:
        print(datasets_df[band_type].iloc[datasets_idx])
    except:
         print(band_type + ' not in file!') 
    

# Write Excel output
datasets_df.to_excel(xls_fullpath)