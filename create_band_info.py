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

                                               
for i_band in range(2): #range(raster_data_array.shape[0]):
    plt.figure()
    plt.imshow(raster_data_array[i_band,:,:]) 
    plt.show()  # display it
    
# Promt user and ask if band list should be stored
# - if there are already band lists, show the current list and the pre-existing one
