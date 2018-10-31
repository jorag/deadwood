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
    def __init__(self, name, **kwargs):
        self.name = name
        # Set default values
        # Class settings
        self.meta_missing_value = None
        self.meta_types = 'meta'
        self.modality_missing_value = np.NaN
        self.modality_types = 'modality'
        # Misc settings
        self.log_type = 'default'
            
        # List of points useful for internal referencing
        self.__last_idx = -1 
        self.idx_list = []
        # Lists of points useful for sorting
        self.point_name = []
        self.point_class = []
        self.point_class_i = [] # Numeric class indicator
        # List of point objects
        self.data_points = []
        
        # Add option to specify  parameter with keywordargs (overwirte default values)
        # May be useful for loading object
        for key, value in kwargs.items():
            setattr(self, key, value)

            
    def split(self, split_type = 'random', train_pct = 0.8, test_pct = 0.2, val_pct = 0, **kwargs):
        # Split in training, validation, and test sets
        n_points = self.__last_idx + 1
        if n_points < 2:
            print('No data to split!')
            return
        # TODO: Input check of split fractions??
        if split_type in ['random', 'rnd']:
            #
            n_train = np.ceil(train_pct*n_points)
            n_val = np.floor(val_pct*n_points)
            # Ensure that number in each set sums to the number of points
            n_test = n_points - n_train - n_val
            
            # Draw training set
            self.set_train = np.random.choice(self.idx_list, size=n_train, replace=False, p=None)
            self.set_train.tolist()
            # Could use p to incorperate relative class probablities, 
            # In that case it should be normalized to 1 before drawing test set
            
            remaining_points = list(set(self.idx_list) - set(self.set_train))
            if n_val == 0:
                self.set_test = remaining_points
                self.set_val = []
            else:
                self.set_test = np.random.choice(remaining_points, size=n_test, replace=False, p=None)
                self.set_test.tolist()
                self.set_val = list(set(remaining_points) - set(self.set_test))
            
            # Loop over all points 
            for i_point in self.set_train:
                self.data_points[i_point].set = 'train'
            
            for i_point in self.set_test:
                self.data_points[i_point].set = 'test'
                
            for i_point in self.set_val:
                self.data_points[i_point].set = 'val'
                
            print('Training: ', self.set_train, '\n')
            print('Testing: ', self.set_test, '\n')
            print('Validation: ', self.set_val, '\n')
            
        
    def add_points(self, point_name, point_class):
        # Check that lengths match
        # TODO: Consider changing so that point_class is not added at creation
        if numel(point_name) != numel(point_class):
            raise AssertionError('DataModalities: Lenght of point names and point classes do not match!', length(point_name), length(point_class))
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
            if not isinstance(point_class, list):
                point_class = make_list(point_class)
        
        # Add data points to DataModalities list
        for i_point in range(length(point_name)):
            # Internal id (index)
            self.__last_idx += 1
            self.idx_list.append(self.__last_idx)
            # Add data points to DataModalities list
            self.point_name.append(point_name[i_point])
            self.point_class.append(point_class[i_point])
            # Create point
            self.data_points.append(DataPoint(self.__last_idx, self))
            # And add class:
            self.data_points[self.__last_idx].update('meta', point_name = point_name[i_point], point_class = point_class[i_point])
            
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
            
        # Make sure there are points
        if self.__last_idx < 0:
            print('Warning! Object contains no data points!')
            return
        
        # Default (missing update)
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
        # Wrapper for add_to_point
        self.add_to_point(point_name, meta_type, point_meta, 'meta')
        
    def add_modality(self, point_name, modality_type, modality_data):
        # Wrapper for add_to_point
        self.add_to_point(point_name, modality_type, modality_data, 'modality')
        
    def read_data_array(self, set_type):
        # Read out dataset as array
        # Read out different arrays by which data is available?
        # I.E. Read out SAR only area or SAR+OPT area
        
        # Check which set we should use
        if set_type is None:
            set_use = self.idx_list
        elif set_type.lower() in ['all', 'tradata']:
            set_use = self.idx_list
        elif set_type.lower() in ['train', 'training']:
            set_use = self.set_train
        elif set_type.lower() in ['test', 'testing']:
            set_use = self.set_train
        elif set_type.lower() in ['val', 'validation']:
            set_use = self.set_val

        # Initialize numpy output array - or read as list?
        
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
    def __init__(self, id, parent, **kwargs):
        # Initialize variables
        self.id = id
        self.meta_missing_value = parent.meta_missing_value
        self.meta_types = parent.meta_types
        self.modality_missing_value = parent.modality_missing_value
        self.modality_types = parent.modality_types
        
        # List of keys
        self.all_keys = []
        self.meta_keys = []
        self.modality_keys = []
        
        # Set assignment (training, validation, test)
        self.set = None
        
        # Add option to specify  parameter with keywordargs (overwirte default values)
        # May be useful for loading object
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def update(self, input_type, **kwargs):
        # Update all values, potentially overwriting real data with null values
        for key, value in kwargs.items():
            setattr(self, key, value) # Move into if test for adding prefix/type to name...
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
            
    def print_point(self):
        # Print
        print('Point id: ', self.id)
        for key in self.all_keys:
            print(key, ' : ', getattr(self, key))

        
        
        
        
       