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
        split(split_type=, train_pct=, test_pct=, val_pct=) - split in training/test/val sets 
        assign_labels(self, class_dict=) - add numeric labels to points by dict with class names+labels
        read_data_array(modalities, set_type) - read data (train/test/val) array with select modalities
        set_log_type(log_type) - set log type for mytools.logit function
        print_points(point_name=) - print points in point_name list
    """
    def __init__(self, name, **kwargs):
        # Two most important attributes, name and path to dataset
        self.name = name
        self.data_paths = dict() # Keys generated when data points are added
        self.classdef_params = dict() # Params for splitting forest class into live/defoliated 
        # Set default values
        # Class settings
        self.meta_missing_value = None
        self.meta_types = 'meta'
        self.modality_missing_value = np.NaN
        self.modality_types = 'modality'
        self.modality_bands = dict()
        self.modality_order = [] # TODO - change band storage implementation to dict with dataset ID as key?
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
        self.class_dict = dict([]) # Use 0 as 'other' class?
        
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
        
        # Input check of split fractions
        if train_pct + test_pct + val_pct != 1:
            logit('Warning! Dataset split fractions /train/test/val) do not sum to 1!', self.log_type)
            pct_sum = train_pct + test_pct + val_pct
            train_pct /= pct_sum 
            test_pct /= pct_sum 
            val_pct /= pct_sum 

        # Switch between different split types
        # Weighted
        if split_type in ['weighted', 'class_weight']:
            # Find the unique labels and counts
            unique_labels = np.unique(self.point_label)

            # Initilaize sets
            self.set_train = []
            self.set_val = []
            self.set_test = []
            # Go though classes and split according to fractions
            for i_label in unique_labels:
                # Get list of indice for points in class (as nparray for indexing)
                idx_list = np.asarray(self.idx_list)
                current_points = idx_list[self.point_label == i_label]
                # Find split fraction for class
                n_points_label = current_points.shape[0]
                # Cast as int due to np.random.choices FutureWarning
                n_train = int(np.floor(train_pct * n_points_label))
                n_val = int(np.floor(val_pct * n_points_label))
                # Ensure that number in each set sums to the number of points
                n_test = int(n_points_label - n_train - n_val)
                
                # Draw training set
                set_train_labels = np.random.choice(current_points, size=n_train, replace=False, p=None)
                self.set_train.extend(set_train_labels.tolist())
                # Remaining points
                remaining_points = list(set(current_points) - set(set_train_labels))
                # Draw test set, remaining points are validation set
                set_test_labels = np.random.choice(remaining_points, size=n_test, replace=False, p=None)
                self.set_test.extend(set_test_labels.tolist())
                set_val_labels = list(set(remaining_points) - set(set_test_labels))
                self.set_val.extend(set_val_labels)
                    
        # Random split types
        elif split_type in ['random', 'rnd', 'unsupervised', 'class_weight_random']:
            if split_type in ['class_weight_random']:
                # Get labels for dataset
                labels = self.read_data_labels(self.idx_list)
                # Get weight according to class occurance to incorperate relative class probablities 
                p_use = get_label_weights(labels)
            else:
                p_use = None
                                
            # Cast as int due to np.random.choices FutureWarning
            n_train = int(np.ceil(train_pct*n_points))
            n_val = int(np.floor(val_pct*n_points))
            # Ensure that number in each set sums to the number of points
            n_test = int(n_points - n_train - n_val)
            
            # Draw training set
            self.set_train = np.random.choice(self.idx_list, size=n_train, replace=False, p=p_use)
            self.set_train.tolist()
            
            # Remaining points
            remaining_points = list(set(self.idx_list) - set(self.set_train))
            if n_val == 0:
                self.set_test = remaining_points
                self.set_val = []
            else:
                if split_type in ['class_weight_random']:
                    # Get weight according to remaining class occurances 
                    # use to incorperate relative class probablities 
                    temp_labels = np.asarray(labels)
                    p_use = get_label_weights(temp_labels[remaining_points])
                else:
                    p_use = None
                # Draw test set, remaining points are validation set
                self.set_test = np.random.choice(remaining_points, size=n_test, replace=False, p=p_use)
                self.set_test.tolist()
                self.set_val = list(set(remaining_points) - set(self.set_test))
            
            # Loop over points in each set, and update set membership
            for i_point in self.set_train:
                self.data_points[i_point].set = 'train'
            
            for i_point in self.set_test:
                self.data_points[i_point].set = 'test'
                
            for i_point in self.set_val:
                self.data_points[i_point].set = 'val'

            
    def add_points(self, point_name):
        # Check that lengths match
        # Check if it is more than one element, if not, make sure it is wrapped in a list
        if numel(point_name) == 1:
            if not isinstance(point_name, list):
                point_name = make_list(point_name)
        
        # Add data points to DataModalities list
        for i_point in range(length(point_name)):
            # Internal id (index)
            self.__last_idx += 1
            self.idx_list.append(self.__last_idx)
            # Add data points to DataModalities list
            self.point_name.append(point_name[i_point])
            # Create DataPoint object and append to data_points list, point has some inheritance from parent DataModalities obj.
            self.data_points.append(DataPoint(self.__last_idx, self))
            # Add dataset ID
            self.data_points[self.__last_idx].update('meta', n_trees = 0)
            
            
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
            print('Warning! Object contains no data points!')
            return
        
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
        
        
    def add_modality(self, point_name, modality_type, modality_data, **kwargs):
        # Wrapper for add_to_point - data modality values
        # Store order of modalities, for getting correct band order etc.
        self.modality_order.append(modality_type)
        # Add to modality_bands - potentially also to other fields?
        for key, value in kwargs.items():
            if key.lower() in 'bands_use': 
                self.modality_bands[modality_type] = value
            else:
                setattr(self, key, value) # TODO: remove this?
        # Add to each point
        self.add_to_point(point_name, modality_type, modality_data, 'modality')
        
        
    def add_tree(self, point_name, row):
        # Wrapper for add_to_point - meta information
        i_point = self.point_name.index(point_name)
        # Get current number of trees
        n_trees = self.data_points[i_point].n_trees
        if n_trees == 0:
            kw_update = dict([['trees', [] ]])
            self.data_points[i_point].update('meta', **kw_update)
        
        kw_update = dict([['n_trees', n_trees+1]])
        self.data_points[i_point].update('meta', **kw_update)
        self.data_points[i_point].trees.append(row)
        
        
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
        
    
    def read_data_array(self, modalities, set_type):
        # Read out dataset as array
        # TODO: Read out different arrays by which data is available? I.E. Read out SAR only area or SAR+OPT area
        
        # Ensure that modalities appear as a list
        if numel(modalities) == 1:
            if not isinstance(modalities, list):
                modalities = make_list(modalities)
        
        # Check which set we should read (training/test/val)
        if set_type is None:
            set_use = self.idx_list
        elif set_type.lower() in ['all', 'data']:
            set_use = self.idx_list
        elif set_type.lower() in ['train', 'training']:
            set_use = self.set_train
        elif set_type.lower() in ['test', 'testing']:
            set_use = self.set_test
        elif set_type.lower() in ['val', 'validation']:
            set_use = self.set_val

        # Initialize numpy output array - or read as list?
        data_array = []
        label_array = []
        # Loop over points in each set, and update set membership
        for i_point in set_use:
            data_array.append(self.data_points[i_point].read_data(modalities)) 
            label_array.append(self.data_points[i_point].label)
        
        # Convert to numpy array
        array_out = np.asarray(data_array)
        return array_out, label_array
        
    
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
        read_data(modalities) - read specified data modalities
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
        
        # List of keys
        self.all_keys = []
        self.meta_keys = []
        self.modality_keys = []
        
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
                    
                    
    def read_data(self, modalities):
        # Read data modalities
        data_out = []
        for key in modalities:
            # Check that key is a valid data modality (else add empty value??) 
            if key in self.modality_keys:  # Use .lower() for improved ruggedness
                # Get values and add to list
                data_add = getattr(self, key)
                if not isinstance(data_add, list):
                    data_add = make_list(data_add)
                data_out += getattr(self, key)
                
        # Return data
        return data_out

            
    def print_point(self):
        # Print
        print('Point id: ', self.id)
        for key in self.all_keys:
            print(key, ' : ', getattr(self, key))
            
            
class Tree:
    """Store information about a single tree.
    
    A typical tree in the study area may have multiple stems, but is still 
    cinsidered a single "functional tree".
    Functions:
        __init__()
     """
    def __init__(self, id, parent, **kwargs):
        # Initialize variables
        self.id = id
        for key, value in kwargs.items():
            setattr(self, key, value)

        
       