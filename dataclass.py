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
        # Initialize variables
        self.name = name
        
        # List of points useful for internal referencing
        self.last_idx = -1
        self.idx_list = []
        # Lists of points useful for sorting
        self.point_name = []
        self.point_class = []
        # List of point objects
        self.data_points = []
        
        # Misc settings
        self.log_type = 'default'
        
    def add_points(self, point_name, point_class, data_modalities=None):
        # Add data points
        if length(data_id) != length(data_class):
            raise AssertionError('DataModalities: Lenght of data ID and data class do not match!', length(data_id), length(data_class))
        
        # Add data points to DataModalities list
        for i_point in range(length(point_name)):
            # Internal id (index)
            self.last_idx += 1
            self.idx_list.append(self.last_id)
            # Add data points to DataModalities list
            self.point_name.append(point_name[i_point])
            self.point_class.append(point_class[i_point])
            # Create point
            self.data_points.append(DataPoint(self.last_idx))

        
    def add_modality(self, modality_name, modality_data):
        # 
        self.last_index
        class_name
        class_number
        class_data
        # Add data modality
        logit('Implement ADD_MODALITY function in DataModalities!', self.log_type)
        
    def read_data_array(self):
        # Read out dataset as array
        # Read out different arrays by which data is available?
        # I.E. Read out SAR only area or SAR+OPT area
        logit('Implement READ_DATA_ARRAY function in DataModalities!', self.log_type)
        return data_array
        
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
        save() - save DataModalities object 
        load() - load DataModalities object
        add_modality() - add new modality
        read_data_array() - read dataset as array
     """
    def __init__(self, id, **kwargs):
        # Initialize variables
        self.id = id
        
     
    def update(self, **kwargs):
        # Initialize variables
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
        
        
       