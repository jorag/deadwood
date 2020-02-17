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
        add_points(point_name, point_class) - add data points
        add_to_point(point_name, update_key, update_value, update_type) - add fields to data points
        add_meta(point_name, meta_type, point_meta) - add meta info to point using add_to_point()
        add_modality(point_name, modality_type, modality_data) - add new modality using add_to_point()
        assign_labels(self, class_dict=) - add numeric labels to points by dict with class names+labels
        read_data_array(modalities, set_type) - read data (train/test/val) array with select modalities
        set_log_type(log_type) - set log type for mytools.logit function
        print_points(point_name=) - print points in point_name list
    """
    def __init__(self, name, **kwargs):
        # Set default values
        self.name = name
        self.data_paths = dict() # Keys generated when data points are added
        self.classdef_params = dict() # Params for splitting forest class into live/defoliated 
        
        # Class settings
        self.meta_missing_value = None
        self.meta_types = 'meta'
        self.modality_missing_value = np.NaN
        self.modality_types = 'modality'
        self.all_modalities = [] # List of all data modalities in object
        self.label_missing_value = -1 # For np.unique to count number of classes
        # Misc settings
        self.log_type = 'default'
            
        # List of points useful for internal referencing
        self.__last_idx = -1 
        self.idx_list = []
        # Lists of points useful for sorting
        self.point_name = []
        self.point_class = []
        self.point_label = [] # Numeric class indicator
        # List of point objects
        self.data_points = []
        
        # Mapping of class names to class numbers as a dict
        self.class_dict = dict([]) 
        
        # Specify extra parameters with keywordargs (overwrite default values)
        for key, value in kwargs.items():
            setattr(self, key, value)

            
    def add_points(self, point_name):
        # Check that lengths match
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
        
        # Add data points to DataModalities list
        for i_point in range(length(point_name)):
            # Internal ID (index)
            self.__last_idx += 1
            self.idx_list.append(self.__last_idx)
            # Add data points to DataModalities list
            self.point_name.append(point_name[i_point])
            # Create DataPoint object and append to data_points list, point has some inheritance from parent DataModalities obj.
            self.data_points.append(DataPoint(self.__last_idx, self))
            
            
    def add_to_point(self, point_name, update_key, update_value, update_type):
        # TODO: Check that update_key is string??
        # Check that lengths match
        if numel(point_name) != numel(update_value):
            raise AssertionError('DataModalities: Lenght of point names and point metadata do not match!', length(point_name), length(update_value))
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
            if not isinstance(update_value, list):
                update_value = make_list(update_value)
            
        # Make sure DataModalities object has points
        if self.__last_idx < 0:
            raise ValueError('Error! Object contains no data points!')
        
        # Default values to use when data is missing (different for data modalities and meta info)
        if update_type in self.meta_types:
            kw_update_missing = dict([[update_key, self.meta_missing_value]])
        elif update_type in self.modality_types:
            kw_update_missing = dict([[update_key, self.modality_missing_value]])
        else:
            raise ValueError('Unkown update_type in DataModalities add_to_point')
        

        # Loop over all points in object and add metadata fields to ALL points,
        # regardless of a value is given or not (omitted points get None as value)
        for i_point in range(self.__last_idx + 1):
            # Check if point should be updated
            if self.point_name[i_point] in point_name:
                # Get value for update ((.index crashes if value is not in list))
                current_val = update_value[point_name.index(self.point_name[i_point])]
                # For passing keyworargs
                kw_update = dict([[update_key, current_val]])
                self.data_points[i_point].update(update_type, **kw_update)
            else:
                # Set default missing value
                self.data_points[i_point].soft_update(update_type, **kw_update_missing)
         
                       
    def add_meta(self, point_name, meta_type, point_meta):
        # Wrapper for add_to_point - meta information
        self.add_to_point(point_name, meta_type, point_meta, 'meta')
        
        
    def add_modality(self, point_name, modality_type, modality_data, sat_data_id, **kwargs):
        # Check that lengths match
        if numel(point_name) != numel(modality_data):
            raise AssertionError('DataModalities: Lenght of point names and point metadata do not match!', length(point_name), length(modality_data))
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
            if not isinstance(modality_data, list):
                modality_data = make_list(modality_data)
            
        # Make sure DataModalities object has points
        if self.__last_idx < 0:
            raise ValueError('Error! Object contains no data points!')
        
        # Add modality type to list of modalities for object
        if not modality_type in self.all_modalities:
            self.all_modalities.append(modality_type)
        
        # Loop over all points in object and add metadata fields to ALL points,
        # regardless of a value is given or not (omitted points get None as value)
        for i_point in range(self.__last_idx + 1):
            # Check if point should be updated
            if self.point_name[i_point] in point_name:
                # Get value for update ((.index crashes if value is not in list))
                current_val = modality_data[point_name.index(self.point_name[i_point])]
                self.data_points[i_point].add_data(sat_data_id, modality_type, modality_data)
            else:
                # Set default missing value
                # TODO: 20190819 - this currently overwrites legit values, make soft update function, or simply skip this step?
                # - maybe just remove this as this function is currently called for one point at a time!
                # self.data_points[i_point].add_data(sat_data_id, modality_type, None)
                pass
        
        
    def add_tree(self, point_name, row, header, exclude_list = []):
        # Find current point
        i_point = self.point_name.index(point_name)
        # Add all values from col to tree
        for col in header:
            # To avoid duplicate fields found in vegetation data file
            # TODO: Consider renaming fields here, and add to tree info anyway?
            if not col in exclude_list: # TODO: Convert both to .lower() ??
                self.data_points[i_point].tree_update(col, getattr(row, col))
        
        # Update current number of trees
        n_trees = self.data_points[i_point].n_trees
        self.data_points[i_point].update('meta', **dict([['n_trees', n_trees+1]]))
        
        
    def assign_labels(self, class_dict = None):
        # Assign numeric labels to data points based on class_dict
        if class_dict is None:
            logit('Warning! No class dict provided, creating based on class names.', self.log_type)
            # No class dict specified, assign each named class its own number
            unique_classes = np.unique(self.point_class)
            # Create dictionary
            class_dict = []
            for i_class in range(length(unique_classes)):
                class_dict.append([unique_classes[i_class], i_class])
                
            self.class_dict = dict(class_dict)
        elif isinstance(class_dict, dict):
            # Set class dict to input (otherwise, ignore and use previously set class dict)
            self.class_dict = class_dict

        # Assign each class in dict a unique number, others to 0 
        self.point_label = [] # reset
        # Get value for other class (if any)
        other_val = self.class_dict.get('other')
        for i_point in self.idx_list:
            val = self.class_dict.get(self.data_points[i_point].point_class)
            if val is not None:
                # Class name found in dict, assign to label given as value
                self.point_label.append(val)
                self.data_points[i_point].label = val
            elif other_val is not None:
                # Class name NOT found in dict, assign to label given for 'other' class
                self.point_label.append(other_val)
                self.data_points[i_point].label = other_val
            else:
                # Class name NOT found in dict and no 'other' class
                self.point_label.append(self.label_missing_value) 
                self.data_points[i_point].label = self.label_missing_value
                    
        # Log and return result
        logit('Number of classes out = ' + str(length(np.unique(self.point_label))), self.log_type)
        return self.point_label, self.class_dict
        
    
    def read_data_points(self, data_type, modality_type=None):
        # Read out data_type of dataset as array
        
        # Initialize numpy output array - or read as list?
        data_array = []
        # Loop over points in each set, and update set membership
        for i_point in self.idx_list:
            if modality_type is None:
                data_array.append(self.data_points[i_point].read_key(data_type))
            else:
                data_array.append(self.data_points[i_point].read_key(data_type, modality_type))
            #label_array.append(self.data_points[i_point].label)
        
        # Convert to numpy array
        array_out = np.asarray(data_array)
        return array_out
        
    
    def read_point(self, point_name, attr):
        # Read an attribute from a single point (used from top level)
        
        # Find point
        i_point = self.point_name.index(point_name)
        return getattr(self.data_points[i_point], attr)
            
    
    def print_points(self, point_name = None):
        # Check if all points or a subset should be printed
        if point_name == None:
            point_name = self.point_name
        
        # Ensure that point names appear as a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
        
        # Add data points to DataModalities list
        for point in point_name:
            # Read from DataPoint class by keeping list of attributes of all points and loop over using getattrb
            self.data_points[self.point_name.index(point)].print_point()
        
        
class DataPoint:
    """Data point.
    
    Functions:
        update(input_type, **kwargs) - set variables to input given in keywordargs
        soft_update(input_type, **kwargs) - as update(), but don't overwrite valid data with None/NaN
        read_key(key) - read specified data field
        print_point() - print all meta and modality keys
     """
    def __init__(self, id, parent, **kwargs):
        # Initialize variables
        self.id = id
        self.meta_missing_value = parent.meta_missing_value
        self.meta_types = parent.meta_types
        self.modality_missing_value = parent.modality_missing_value
        self.modality_types = parent.modality_types
        self.label = None # Numeric label 
        self.n_trees = 0 # Number of trees
        # List of keys - TODO: Consider a 'tree' key
        self.all_keys = []
        self.meta_keys = ['n_trees']
        self.modality_keys = []
        self.data_keys = []
        
        # Set assignment (training, validation, test)
        self.set = None
        
        # Add option to specify  parameter with keywordargs (overwrite default values)
        # May be useful for loading object
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
    def update(self, input_type, **kwargs):
        # Update all values, potentially overwriting real data with null values
        for key, value in kwargs.items():
            setattr(self, key, value) 
            if not key in self.all_keys:
                self.all_keys.append(key)
                if input_type in self.meta_types:
                    self.meta_keys.append(key)
                if input_type in self.modality_types:
                    self.modality_keys.append(key)
                
                
    def soft_update(self, input_type, **kwargs):
        # Only update values with values different from missing values
        for key, value in kwargs.items():
            if key in self.all_keys:
                # Check if default missing value
                if input_type in self.meta_types and value != self.meta_missing_value and np.isnan(value) == False:
                    setattr(self, key, value)
                if input_type in self.modality_types and value != self.modality_missing_value and np.isnan(value) == False:
                    setattr(self, key, value)
            else:
                self.all_keys.append(key)
                setattr(self, key, value)
                if input_type in self.meta_types:
                    self.meta_keys.append(key)
                if input_type in self.modality_types:
                    self.modality_keys.append(key) 
    
    
    def add_data(self, sat_data_id, update_type, update_value):
        # Get previous values (if any)
        try:
            data_dict = getattr(self, sat_data_id)
            # Add new value to dict
            data_dict[update_type] = update_value
        except:
            data_dict = dict([[update_type, update_value]])
        
        # Update dict
        setattr(self, sat_data_id, data_dict) 
        
        # Update keys
        if not sat_data_id in self.all_keys:
            self.all_keys.append(sat_data_id)
            self.data_keys.append(sat_data_id)
    
    
    def tree_update(self, col, val):
        # Update list of tree measurements
        if self.n_trees == 0: # Initialize, to store keys in list (for printing)
            #setattr(self, col, []) # To "hide" key to avoid printing
            self.update('meta', **dict([[col, [] ]]))
        
        # Append measurement to list
        curr_list = getattr(self, col)
        curr_list.append(val)
        setattr(self, col, curr_list)


    def read_key(self, key1, key2=None): 
        value = getattr(self, key1)
        if key2 is not None:
            # Return data
            return value[key2]
        else:
            return value

            
    def print_point(self):
        # Print
        print('Point id: ', self.id)
        for key in self.all_keys:
            print(key, ' : ', getattr(self, key))
            
