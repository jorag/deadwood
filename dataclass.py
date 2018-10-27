#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

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
    def __init__(self, name, data_id):
        # Initialize variables
        self.name = name
        self.data_id = data_id
        self.log_type = 'default'
        
    def add_modality(self, modality_name, modality_data):
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
       