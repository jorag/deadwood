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
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *

# Classify LIVE FOREST vs. DEFOLIATED FOREST vs. OTHER

# Name of input object and file with satellite data path string
dataset_use = 'vanZyl-A'
obj_in_name = dataset_use + '.pkl'
sat_pathfile_name = dataset_use + '-path'

# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

# Load DataModalities object
with open(os.path.join(dirname, "data", obj_in_name), 'rb') as input:
    input_data = pickle.load(input)
    

# TRAIN AND CROSS-VALIDATE

# Set class labels for dictionary
class_dict_in = dict([['Live', 1], ['Defoliated', 2], ['other', 0]])

# Get labels and class_dict (in case None is input, one is created)
labels, class_dict = input_data.assign_labels(class_dict=class_dict_in)


# Split into training, validation, and test sets
input_data.split(split_type = 'weighted', train_pct = 0.9, test_pct = 0.1, val_pct = 0.0)
print(length(input_data.set_train)/165, length(input_data.set_test)/165, length(input_data.set_val)/165)

# Get all data
all_data, all_labels = input_data.read_data_array(['quad_pol', 'optical'], 'all') 
# Get SAR data
sar_data, sar_labels = input_data.read_data_array(['quad_pol'], 'all') 

# Normalize data - should probably be done when data is stored in object...
print(np.max(all_data,axis=0))

# Split into training and test datasets
data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=0)  

# Create kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
# Fit kNN
neigh.fit(data_train, labels_train) 

# Score kNN
print(neigh.score(data_test, labels_test)) 
# Test kNN on test dataset
prediction_result = neigh.predict(data_test)

# Cross validate - All data
knn_all = KNeighborsClassifier(n_neighbors=3)
scores_all = cross_val_score(knn_all, all_data, all_labels, cv=5)
print(scores_all) 

# Cross validate - SAR data
knn_sar = KNeighborsClassifier(n_neighbors=3)
scores_sar = cross_val_score(knn_sar, sar_data, sar_labels, cv=5)
print(scores_sar) 


# Test Random Forest

rf = RandomForestClassifier(n_estimators=20, random_state=0)  
rf.fit(data_train, labels_train) 
y_pred = rf.predict(data_test) 
print(rf.score(data_test, labels_test)) 


# TEST ON COMPLETE IMAGE

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
#    # Get single layer list
#    for item in input_data.modality_bands[key]:
#        bands_use_single.append(item[0])

## Predict class for entire satellite image
sat_im = all_sat_bands[bands_use_single , :, : ]
## Reshape array to n_cols*n_rows rows with the channels as columns 
sat_im_prediction, n_rows, n_cols = imtensor2array(sat_im)

# NORMALIZE IMAGE - TODO: Change to 'local' both here and in get_sat_data??
sat_im_prediction = norm01(sat_im_prediction, norm_type='global', log_type = 'print')

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
plt.show()  # display it




# RF image
rf_im_result = rf.predict(sat_im_prediction)
# Reshape to original input size
sat_result_rf = np.reshape(rf_im_result, (n_rows, n_cols))

fig2 = plt.figure()
plt.imshow(sat_result_rf.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
plt.colorbar(ticks=np.unique(list(class_dict.values())) )
plt.show()  # display it
