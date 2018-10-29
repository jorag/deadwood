#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np
from mytools import *

class DataModalities:
    """Create DataModalities object for training data.
    
    Functions:
        save() - save DataModalities object 
        load() - load DataModalities object
        add_modality() - add new modality
        read_data_array() - read dataset as array
        set_log_type(log_type) - set log type for mytools.logit function 
    """
    def __init__(self, name):
        # Class settings
        self.name = name
        self.meta_missing_value = None
        self.modality_missing_value = np.NaN
        
        # List of points useful for internal referencing
        self.last_idx = -1 # TODO: Make private??
        self.idx_list = []
        # Lists of points useful for sorting
        self.point_name = []
        self.point_class = []
        # List of point objects
        self.data_points = []
        
        # Misc settings
        self.log_type = 'default'
        
    def add_points(self, point_name, point_class):
        # Check that lengths match
        if numel(point_name) != numel(point_class):
            raise AssertionError('DataModalities: Lenght of point names and point classes do not match!', length(point_name), length(point_class))
        
        # Add data points to DataModalities list
        for i_point in range(length(point_name)):
            # Internal id (index)
            self.last_idx += 1
            self.idx_list.append(self.last_idx)
            # Add data points to DataModalities list
            self.point_name.append(point_name[i_point])
            self.point_class.append(point_class[i_point])
            # Create point
            self.data_points.append(DataPoint(self.last_idx))
            # And add class:
            self.data_points[self.last_idx].update(point_class = point_class[i_point])
            
    def add_meta(self, point_name, meta_type, point_meta):
        # TODO: Check that meta_type is string??
        # TODO: Check if only one input (use length??), and wrap in list if True??
        # Check that lengths match
        if numel(point_name) != numel(point_meta):
            raise AssertionError('DataModalities: Lenght of point names and point metadata do not match!', length(point_name), length(point_meta))
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
            if not isinstance(point_meta, list):
                point_meta = make_list(point_meta)
            
        # Make sure there are points
        if self.last_idx < 0:
            print('Warning! Object contains no data points!')
            return
        
        # Default (missing update)
        kw_update_missing = dict([[meta_type, self.meta_missing_value]])
        
        # Loop over all points in object and add metadata fields to ALL points,
        # regardless of a value is given or not (omitted points get None as value)
        for i_point in range(self.last_idx + 1):
            # Check if point should be updated
            if self.point_name[i_point] in point_name:
                # Get value for update ((.index crashes if value is not in list))
                meta_val = point_meta[point_name.index(self.point_name[i_point])]
                # For passing keyworargs
                kw_update = dict([[meta_type, meta_val]])
                self.data_points[i_point].update(**kw_update)
            else:
                # Set default missing value
                self.data_points[i_point].soft_update(**kw_update_missing)


        
    def add_modality(self, modality_name, modality_data):
        # Add data modality
        logit('Implement ADD_MODALITY function in DataModalities!', self.log_type)
        
    def read_data_array(self):
        # Read out dataset as array
        # Read out different arrays by which data is available?
        # I.E. Read out SAR only area or SAR+OPT area
        logit('Implement READ_DATA_ARRAY function in DataModalities!', self.log_type)
        return data_array
        
    def print_points(self, point_name = None):
        # Check if all points or a subset should be printed
        if point_name == None:
            point_name = self.point_name
            
        # Add data points to DataModalities list
        for point in point_name:
            # Read from DataPoint class by keeping list of attributes of all points and loop over using getattrb
            #print(point)
            #print(self.point_class[self.point_name.index(point)])
            self.data_points[self.point_name.index(point)].print_point()
            #curr_point = self.point_class[self.point_name.index(point)]
            #curr_point.print_point()
            #print(np.where(self.point_name == point))
            
    def save(self, filename):
        # Save
        logit('Implement SAVE function in DataModalities!', self.log_type)
        
    def load(self, filename):
        # Load
        logit('Implement LOAD function in DataModalities!', self.log_type)
        
    def set_log_type(self, log_type):
        # Change log type
        self.log_type = log_type
        
        
class DataPoint:
    """Data point.
    
    Functions:
        update(**kwargs) - set variables to input given in keywordargs
     """
    def __init__(self, id, **kwargs):
        # Initialize variables
        self.id = id
        self.keys = []
        
    def update(self, **kwargs):
        # Initialize variables
        # TODO: DO NOT UPDATE MISSING VALUES (WRITING NONE/NAN)
        for key, value in kwargs.items():
            setattr(self, key, value)
            if not key in self.keys:
                self.keys.append(key)
                
    def soft_update(self, **kwargs):
        # Initialize variables
        # TODO: Get different missing update values (np.nan)
        for key, value in kwargs.items():
            if key in self.keys:
                # Check if default missing value
                if value != None and np.isnan(value) == False:
                    setattr(self, key, value)
            else:
                self.keys.append(key)
                setattr(self, key, value)
            
    def print_point(self):
        # Print
        print('Point id: ', self.id)
        for key in self.keys:
            print(key, ' : ', getattr(self, key))

        
        
        
        
       