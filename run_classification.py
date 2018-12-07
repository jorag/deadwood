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
obj_in_name = "obj-C.pkl"

dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER

# Load DataModalities object
with open(os.path.join(dirname, "data", obj_in_name), 'rb') as input:
    input_data = pickle.load(input)
    

# Read satellite data
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
class_dict = dict([['Forest', 1], ['Wooded mire', 2], ['other', 0]])
#class_dict = None
labels = input_data.assign_labels(class_dict=class_dict)

# Split into training, validation, and test sets
input_data.split(split_type = 'weighted', train_pct = 0.9, test_pct = 0.1, val_pct = 0.0)
print(length(input_data.set_train)/165, length(input_data.set_test)/165, length(input_data.set_val)/165)


# Create kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)

# Get training data
# TODO: Implement an 'all' option for modalities
data_train, labels_train = input_data.read_data_array('quad_pol', 'train') 
# Fit kNN
neigh.fit(data_train, labels_train) 

# Get test data
# TODO: Implement an 'all' option for modalities
data_test, labels_test = input_data.read_data_array('quad_pol', 'test') 
# Score kNN
print(neigh.score(data_test, labels_test)) 
# Test kNN
prediction_result = neigh.predict(data_test) 


# Predict class for satellite image
sat_im = all_sat_bands[0:3, :, :]
n_channels = sat_im.shape[0]
n_rows = sat_im.shape[1]
n_cols = sat_im.shape[2]
# Reshape array to n_cols*n_rows rows with the channels as columns 
sat_im = np.transpose(sat_im, (1, 2, 0)) # Change order to rows, cols, channels
sat_im2 = np.reshape(sat_im, (n_rows*n_cols, n_channels))
sat_im_result = neigh.predict(sat_im2)


# Show entire image
plt.figure()
plt.imshow(sat_im) 
plt.show()  # display it


# Reshape to original input size
sat_result2 = np.reshape(sat_im_result, (n_rows, n_cols))
# Show classification result
colors = ['red','green','blue','purple']
cmap = plt.get_cmap('jet', 3)
fig = plt.figure()
#plt.imshow(sat_result2, cmap='jet')
plt.imshow(sat_result2.astype(int), cmap=cmap, vmin=-0.5, vmax=2.5)
plt.colorbar(extend='min')
#plt.colorbar()
plt.show()  # display it
#plt.imshow(sat_result2, cmap=matplotlib.colors.ListedColormap(colors))
