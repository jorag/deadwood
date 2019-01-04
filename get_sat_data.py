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
import pandas as pd
import os # Necessary for relative paths
import xml.etree.ElementTree as ET
import pickle
#import sys # To append paths
# My moduels
from mytools import *
from geopixpos import *
from visandtest import *
from dataclass import *


# Set name of output object
obj_out_name = "dataset-B.pkl"

dirname = os.path.realpath('.') # For parent directory use '..'

# Classify LIVE FOREST vs. DEAD FOREST vs. OTHER


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
            

# Get georeference info
geotransform = dataset.GetGeoTransform()
    
## Load a band, 1 based
#band = dataset.GetRasterBand(1)
#bandinfo_log(band, log_type='default')

# Read multiple bands
all_sat_bands = dataset.ReadAsArray()

# Dataset: A lat = 45, lon = 46. Dataset B & C: lat = 21, lon = 22
lat_band = dataset.GetRasterBand(21)
lon_band = dataset.GetRasterBand(22)

#showimpoint(all_sat_bands, geotransform, point_lat, point_lon, n_pixel_x=500, n_pixel_y=500, bands=[0,1,2])
    

# Read Excel file with vegetation types
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "vegetation-data-path")) as infile:
        veg_file = infile.readline().strip()
        logit('Read file: ' + veg_file, log_type = 'default')
    
    # Load data
    xls_veg = pd.ExcelFile(veg_file)
except:      
    logit('Error, promt user for file.', log_type = 'default')
    # Predefined file failed for some reason, promt user
    root = tkinter.Tk() # GUI for file selection
    root.withdraw()
    veg_file = tkinter.filedialog.askopenfilename(title='Select input .csv/.xls(x) file')
    xls_veg = pd.ExcelFile(veg_file)


# Go through all sheets in Excel file for vegetation
point_info = []
class_veg = []
name_veg = []
for i_sheet in range(1,7):
    print(i_sheet)
    # Get pandas dataframe
    df = pd.read_excel(xls_veg, str(i_sheet))
    point_id = list(df['GPSwaypoint'])
    # Go through the list of points
    for id in point_id:
        name_veg.append(id) # Point name, e.g. 'N_6_159'
        class_veg.append(df['LCT1_2017'][point_id.index(id)]) # Terrain type, e.g. 'Forest'


# Read .gpx file with coordinates of transect points
try:
    # Read predefined file
    with open(os.path.join(dirname, "data", "gps-data-path")) as infile:
        gps_file = infile.readline().strip()
        logit('Read file: ' + gps_file, log_type = 'default')
    
    # Load data
    tree = ET.parse(gps_file)
except:      
    logit('Error, promt user for file.', log_type = 'default')
    # Predefined file failed for some reason, promt user
    root = tkinter.Tk() # GUI for file selection
    root.withdraw()
    gps_file = tkinter.filedialog.askopenfilename(title='Select input .gpx file')
    # Load data
    tree = ET.parse(gps_file)

# Get lat and long
pos_array = []
for elem in tree.findall("{http://www.topografix.com/GPX/1/1}wpt"):
    lon, lat = elem.attrib['lon'], elem.attrib['lat']
    pos_array.append((float(lat), float(lon)))
# Get name of waypoints
gps_id = []
for elem in tree.findall("//{http://www.topografix.com/GPX/1/1}name"):
    gps_id.append(elem.text)


# Read Excel file with tree data
try:
    # Read predefined tree data file
    with open(os.path.join(dirname, "data", "tree-data-path")) as infile:
        tree_file = infile.readline().strip()
        logit('Read file: ' + tree_file, log_type = 'default')
    
    # Load data
    xls_tree = pd.ExcelFile(tree_file)
except:      
    logit('Error, promt user for file.', log_type = 'default')
    # Predefined file failed for some reason, promt user
    root = tkinter.Tk() # GUI for file selection
    root.withdraw()
    tree_file = tkinter.filedialog.askopenfilename(title='Select input .csv/.xls(x) file')
    xls_tree = pd.ExcelFile(tree_file)


# Store class as dict with GPSwaypoint as ID
class_dict = dict(zip(gps_id, class_veg))

# Initialize output lists and temporary variables
point_info = []
class_tree = []
name_tree = []
lai_point = []
prev_id = 'dummy'
# Go through all sheets in Excel sheet
for i_sheet in range(1,7):
    print(i_sheet)
    # Get pandas dataframe
    df = pd.read_excel(xls_tree, str(i_sheet))
    point_id = list(df['ID']) # Country
    # Go through the list of points
    for row in df.itertuples(index=True, name='Pandas'):
        # Check if the current tree is in a new transect point
        curr_id = str(row.Country) + '_' + str(row.Transect) + '_'  + str(row.ID)
        if curr_id != prev_id:
            print(prev_id)
            print(row.Stemdiam2, getattr(row, 'Stemdiam3'))
            prev_id = curr_id
            LAI_proxy_temp = 0
            # Add waypoint name to list of IDs
            name_tree.append(curr_id)
            # Add LAI to list or use to classify point as is???
            lai_point.append(0)
        
        # Add area of crown (based on DIAMETERS) to last point. (ellipse*fraction_live)/total_point_area
        lai_point[-1] += np.pi/4 * row.Crowndiam1 * row.Crowndiam2 * row.CrownPropLive/100
        #LAI_proxy_temp += np.pi/4 * row.Crowndiam1 * row.Crowndiam2 * row.CrownPropLive/100
        #print(row.Crowndiam1, getattr(row, 'Crowndiam2'), row.Treeheight)

# Store Leaf Area Index and names as array
lai_array = np.asarray(lai_point)
tree_point_array = np.asarray(name_tree)
live_crown_points = tree_point_array[lai_array>0]

# Sort into healthy and defoliated forest
lai_threshold_live = 1.5
for i_point in range(length(name_tree)):
    # Check if Leaf Area Index is greater than threshold
    if lai_point[i_point] > lai_threshold_live:
        print(class_dict[name_tree[i_point]] , '-> Live')
        class_dict[name_tree[i_point]] = 'Live'
    else:
        print(class_dict[name_tree[i_point]] , '-> Defoliated')
        class_dict[name_tree[i_point]] = 'Defoliated'

# Return original order of points
c = [class_dict[x] for x in gps_id]



# Merge names and positions
gps_points = list(zip(gps_id, pos_array))
# Convert to numpy array
#pos_array2 = np.asarray(pos_array)
pos_array2 = np.asarray([item[1] for item in gps_points])
gps_id2 = [item[0] for item in gps_points]


## Intialize data object
# TODO - add meta information here as kwargs, such as year, area, etc.
all_data = DataModalities('Polmak-2017-B')
# Add points
all_data.add_points(name_veg, class_veg, dataset_id = 'SAR-B', dataset_path = sat_file)
# Add GPS points
all_data.add_meta(gps_id, 'gps_coordinates', pos_array)


# Get pixel positions from my geopixpos module
# TODO: Change so that coordinates can be input as tuples
# Dataset: A lat = 45, lon = 46. Dataset B & C: lat = 21, lon = 22
pix_lat, pix_long = geocoords2pix(lat_band.ReadAsArray(), lon_band.ReadAsArray(), lon=pos_array2[:,1], lat=pos_array2[:,0], pixels_out = 'npsingle')

# Extract pixels from area - SAR
# t11 = 11, t22 = 16, t33 = 19
sar_bands_use = [[11], [16], [19]]
data_out = all_sat_bands[sar_bands_use , [pix_lat.T], [pix_long.T]] # Works, gives (3,165) array

# Transpose so that rows correspond to observations
if data_out.shape[0] != length(pix_lat) and data_out.shape[1] == length(pix_lat):
    data_out = data_out.T

# SAR info to add to object
kw_sar = dict([['bands_use', sar_bands_use]])

# Add SAR modality
all_data.add_modality(gps_id, 'quad_pol', data_out.tolist(), **kw_sar)


# Extract pixels from area - OPTICAL
#showimage(np.squeeze(all_sat_bands[[[2], [3], [4]], :, :])/10000, bands=[1,2,0])
opt_bands_use = [[2], [3], [4]]
opt_out = all_sat_bands[opt_bands_use, [pix_lat.T], [pix_long.T]] # Works, gives (3,165) array

# Transpose so that rows correspond to observations
if opt_out.shape[0] != length(pix_lat) and opt_out.shape[1] == length(pix_lat):
    opt_out = opt_out.T

# OPT info to add to object
kw_opt = dict([['bands_use', opt_bands_use]])

# Add OPT modality
all_data.add_modality(gps_id, 'optical', opt_out.tolist(), **kw_opt)


## Print points
#all_data.print_points()

# Set class labels for dictionary
class_dict = dict([['Forest', 1], ['Wooded mire', 2], ['other', 0]])
class_dict = None
labels = all_data.assign_labels(class_dict=class_dict)

# Split into training, validation, and test sets
all_data.split(split_type = 'weighted', train_pct = 0.7, test_pct = 0.3, val_pct = 0.0)

# Save DataModalities object
with open(os.path.join(dirname, "data", obj_out_name), 'wb') as output:
    pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)
