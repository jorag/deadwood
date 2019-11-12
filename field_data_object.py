#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:50:21 2019
@author: jorag
"""

import matplotlib.pyplot as plt
import numpy as np
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
obj_out_name = 'DiffGPS_FIELD_DATA'

# PARAMETERS
# Ground truth info - TODO: Store this info and parameters in object!!
transect_point_area = 10*10 # m^2 (10 m X 10 m around centre of point was examined)

# Processing parameters - minimum of X * 100 m^2 plc to be in 'Live' class
# TODO: NEED TO ADJUST THESE THRESHOLDS AFTER DISCUSSION WITH ECOLOGISTS/EXPERTS
plc_min_live = 0.03 # min Leaf Area Index to be assigned to Live class 
maxstem_min_defo = 2.5 # min registered max stem thickness for defoliated class
ntrees_min_defo = 3 # min number of trees for defoliated class

## Intialize data object
all_data = DataModalities('Field data only')


# Path to working directory 
dirname = os.path.realpath('.') # For parent directory use '..'

                          
# READ GROUND TRUTH DATA FILES
# Read Excel file with vegetation types
try:
    # Read predefined file
    with open(os.path.join(dirname, 'input-paths', 'vegetation-data-path')) as infile:
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
class_veg = []
name_veg = []
for i_sheet in range(1,7):
    # Get pandas dataframe, all IDs and column (header)
    df = pd.read_excel(xls_veg, str(i_sheet))
    point_id = list(df['GPSwaypoint'])
    veg_header = list(df.columns.values)
    # Go through the list of points
    for id in point_id:
        name_veg.append(id) # Point name, e.g. 'N_6_159'
        all_data.add_points(id) # Create point with name, e.g. 'N_6_159'
        class_veg.append(df['LCT1_2017'][point_id.index(id)]) # Terrain type, e.g. 'Forest'
        # Add info to point
        for attr in veg_header:
            # TODO: Consider "translating" some column names using a dict
            all_data.add_to_point(id, attr, [df[attr][point_id.index(id)]], 'meta')


# Read .txt differentian GPS file with coordinates of transect points
with open(os.path.join(dirname, 'input-paths', 'DiffGPS-data-path')) as infile:
    diff_gps_file = infile.readline().strip()
    logit('Read file: ' + diff_gps_file, log_type = 'default')
    
# Load data
diff_gps = pd.read_csv(diff_gps_file, sep='\t')
# Contains point names - create list of point names
diff_gps_points = list(diff_gps['Comment']) 

# DiffGPS - # Get lat and long
dpos_array = []
for point_name in name_veg:
    # Transform transect point name from Excel file name format to DiffGPS name format
    curr_name = point_name[0]+point_name[2]+'-'+point_name[4:]
    lat = diff_gps['Latitude'][diff_gps_points.index(curr_name)]
    lon = diff_gps['Longitude'][diff_gps_points.index(curr_name)]
    dpos_array.append((float(lat), float(lon)))
    print(curr_name, lat, lon)


# Read .gpx file with coordinates of transect points
try:
    # Read predefined file
    with open(os.path.join(dirname, 'input-paths', 'gps-data-path')) as infile:
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
for elem in tree.findall('{http://www.topografix.com/GPX/1/1}wpt'):
    lon, lat = elem.attrib['lon'], elem.attrib['lat']
    pos_array.append((float(lat), float(lon)))

# Get name of waypoints
gps_id = []
for elem in tree.findall('//{http://www.topografix.com/GPX/1/1}name'):
    gps_id.append(elem.text)


# Read Excel file with tree data
try:
    # Read predefined tree data file
    with open(os.path.join(dirname, 'input-paths', 'tree-data-path')) as infile:
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

# Read tree data
# Initialize output lists and temporary variables
class_tree = []
name_tree = []
plc_point = [] # Proportion Live Canopy (PLC) 
pdc_point = [] # Proportion Defoliated Canopy (PDC) (tree crown area without leaves)
n_stems_live = []
n_stems_dead = []
max_stem_thick = []
avg_tree_height = [0]
n_trees = [1]
prev_id = 'dummy'
# Go through all sheets in Excel file with tree data
for i_sheet in range(1,7):
    # Get pandas dataframe
    df = pd.read_excel(xls_tree, str(i_sheet))
    point_id = list(df['ID']) # Country
    # Go through the list of points
    for row in df.itertuples(index=True, name='Pandas'):
        # Get ID of current point 
        # TODO: 20190823 - CHANGE THIS FOR 2019 DATA??
        curr_id = str(row.Country) + '_' + str(row.Transect) + '_'  + str(row.ID)
        
        # New version 
        tree_header = list(df.columns.values)
        all_data.add_tree(curr_id, row, tree_header, exclude_list=veg_header)
        
        # Check if the current tree is in a new transect point
        if curr_id != prev_id:
            # TODO: 20190823 - add plc here?
            if prev_id is not 'dummy':
                all_data.add_to_point(prev_id, 'plc', plc_point[-1], 'meta')
                all_data.add_to_point(prev_id, 'pdc', pdc_point[-1], 'meta')
                all_data.add_to_point(prev_id, 'max_stem_thick', max_stem_thick[-1], 'meta')
                all_data.add_to_point(prev_id, 'avg_tree_height', avg_tree_height[-1], 'meta')
                all_data.add_to_point(prev_id, 'n_stems_live', n_stems_live[-1], 'meta')
                all_data.add_to_point(prev_id, 'n_stems_dead', n_stems_dead[-1], 'meta')
                
            # Add waypoint name to list of IDs
            name_tree.append(curr_id)
            # Calculate average tree height
            avg_tree_height[-1] = avg_tree_height[-1]/n_trees[-1]
            # Add new plc to list
            plc_point.append(0)
            # Add new "pdc" to list
            pdc_point.append(0)
            # Add new Stems_Live to list
            n_stems_live.append(0)
            # Add new Stems_Dead to list
            n_stems_dead.append(0)
            # Add new maximum stem thickness
            max_stem_thick.append(0)
            # Add new average tree height
            avg_tree_height.append(0)
            # Add new tree count
            n_trees.append(0)
            # Keep adding trees to curren point
            prev_id = curr_id
        
        # Add area of crown (based on DIAMETERS in m) to last point (ellipse*fraction_live)/total_point_area
        plc_point[-1] += (np.pi/4*row.Crowndiam1*row.Crowndiam2*row.CrownPropLive/100)/transect_point_area
        # Add area of defoliated crown (DIAMETERS in m) to last point (ellipse*fraction_dead)/total_point_area
        pdc_point[-1] += (np.pi/4*row.Crowndiam1*row.Crowndiam2*(100-row.CrownPropLive)/100)/transect_point_area
        # Update number of live stems
        n_stems_live[-1] += row.Stems_Live
        # Update number of dead stems
        n_stems_dead[-1] += row.Stems_Dead
        # Maximum stem thickness
        max_stem_thick[-1] = np.nanmax([max_stem_thick[-1], row.Stemdiam1, row.Stemdiam2, row.Stemdiam3])
        # Update number of trees count
        n_trees[-1] += 1
        # Sum tree height (divide by final tree count later)
        avg_tree_height[-1] += row.Treeheight

# Remove first (dummy) elements used to avoid throw-away counting variables
n_trees.pop(0)
avg_tree_height.pop(0)


# Merge transect waypoint names and positions
gps_points = list(zip(gps_id, pos_array))
# Convert to numpy array
#pos_array2 = np.asarray(pos_array)
pos_array2 = np.asarray([item[1] for item in gps_points])
gps_id2 = [item[0] for item in gps_points]

# Add GPS points
# 20191111: TODO - Add DiffGPS points here, rename original waypoint_lat/lon?
all_data.add_meta(gps_id, 'gps_coordinates', pos_array)


## Print points
all_data.print_points(['N_4_89', 'N_4_90'])

# Save DataModalities object
with open(os.path.join(dirname, 'data', obj_out_name+'.pkl'), 'wb') as output:
    pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)


