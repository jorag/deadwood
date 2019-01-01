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
from sklearn.neighbors import KNeighborsClassifier
import pickle
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Name of input object
obj_in_name = "dataset-B.pkl"

dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER

# Load DataModalities object
with open(os.path.join(dirname, "data", obj_in_name), 'rb') as input:
    input_data = pickle.load(input)
    

## Read satellite data
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "sat-data-path")) as infile:
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

# TODO: Check why rerunning these commands causes index out of bounds in split (l 84)
# Set class labels for dictionary
class_dict_in = dict([['Forest', 1], ['Wooded mire', 2], ['other', 0]])
class_dict_in = None

# Get labels and class_dict (in case None is input, one is created)
labels, class_dict = input_data.assign_labels(class_dict=class_dict_in)

# Get the list of unique class numbers  
class_n_unique = np.unique(list(class_dict.values()))
# Use the highest and lowest class n for colourbar visualization
class_n_lowest = np.min(class_n_unique) 
class_n_highest = np.max(class_n_unique) 

# Split into training, validation, and test sets
input_data.split(split_type = 'weighted', train_pct = 0.9, test_pct = 0.1, val_pct = 0.0)
print(length(input_data.set_train)/165, length(input_data.set_test)/165, length(input_data.set_val)/165)

# Get training data
# TODO: Implement an 'all' option for modalities
data_train, labels_train = input_data.read_data_array(['quad_pol', 'optical'], 'train') 

# Get test data
# TODO: Implement an 'all' option for modalities
data_test, labels_test = input_data.read_data_array(['quad_pol', 'optical'], 'test') 

# Create kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
# Fit kNN
neigh.fit(data_train, labels_train) 

# Score kNN
print(neigh.score(data_test, labels_test)) 
# Test kNN on test dataset
prediction_result = neigh.predict(data_test) 


# Get bands used in two lists
bands_use_lists = []
bands_use_single = []
for key in input_data.modality_bands.keys():
    # Get list of lists of bands
    bands_use_lists += input_data.modality_bands[key]
    # Get single layer list
    for item in input_data.modality_bands[key]:
        bands_use_single.append(item[0])

## Predict class for entire satellite image
sat_im = all_sat_bands[bands_use_single , :, : ]
# For some reason  [[2], [3], [4], [11], [16], [19]], :,: - gives size (6,1,2892,4182)...
n_channels = sat_im.shape[0]
n_rows = sat_im.shape[1]
n_cols = sat_im.shape[2]
# Reshape array to n_cols*n_rows rows with the channels as columns 
sat_im = np.transpose(sat_im, (1, 2, 0)) # Change order to rows, cols, channels
sat_im_prediction = np.reshape(sat_im, (n_rows*n_cols, n_channels))
sat_im_result = neigh.predict(sat_im_prediction)


# Show entire image
plt.figure()
plt.imshow(sat_im[:,:,[1,2,0]]/16000) 
plt.show()  # display it


# Reshape to original input size
sat_result2 = np.reshape(sat_im_result, (n_rows, n_cols))
# Show classification result
colors = ['red','green','blue','purple']
cmap = plt.get_cmap('jet', length(class_n_unique)) # Number of colours = n. of classes
fig = plt.figure()
#plt.imshow(sat_result2, cmap='jet')
plt.imshow(sat_result2.astype(int), cmap=cmap, vmin=class_n_lowest-0.5, vmax=class_n_highest+0.5)
plt.colorbar(ticks=np.unique(list(class_dict.values())) )
plt.show()  # display it

