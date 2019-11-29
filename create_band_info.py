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
import pickle
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Open Spreadsheat
dataset_xls = pd.ExcelFile(os.path.join(dirname, 'input-paths', '2019_reprocess_dataset_overview.xls'))
dataset_df = pd.read_excel(dataset_xls)
paths_in = list(dataset_df['Path']) 

# Read LAST line - OR index specific dataset
latest_dataset_path = paths_in[-1] 

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
# - ask user if he wants to add band to a list (lat, lon, SAR, or opt), multiple choise

                                               
#for i_band in range(raster_data_array.shape[0]):
#    plt.figure()
#    plt.imshow(raster_data_array[i_band,:,:]) 
#    plt.show()  # display it
             
# Initialize output lists
sorting_dict = {'sar_bands': [], 'opt_bands': [], 'lat_band': [], 
'lon_band': [], 'unsure': [], 'none': []}
#sar_bands = []             
#opt_bands = []
#lat_band = []
#lon_band = []

# List for question dialogue


# Rearrage dimensions to x,y,channel format
im_generate = np.transpose(raster_data_array, (1,2,0))
    
for i_band in range(im_generate.shape[2]-20):
    # Get statistics
    band_min = np.nanmin(im_generate[:,:,i_band])
    band_mean = np.nanmean(im_generate[:,:,i_band])
    band_max = np.nanmax(im_generate[:,:,i_band])
    # Band title text
    band_txt = 'BAND '+ str(i_band) 
    plt.figure(i_band)
    plt.imshow(im_generate[:,:,i_band])
    plt.title(band_txt +' Min = '+str(band_min)+' Mean = '+str(band_mean)+' Max = '+str(band_max))
    plt.pause(0.05)
    #plt.show()  # display it
    
    # Ask which band it its
    which_list = ask_multiple_choice_question('Which band type is this?', list(sorting_dict.keys()), title=band_txt)
    print("User's response was: {}".format(repr(which_list)))
    print(which_list)
    
    # Store result
    sorting_dict[which_list].append(i_band)
    
    plt.close(i_band)
    
# Promt user and ask if band list should be stored
# - if there are already band lists, show the current list and the pre-existing one
