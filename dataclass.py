#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

from mytools import *

class DataModalities:
    """Create DataModalities object.
    
    Functions:
        save() - save DataModalities object 
        load() - load DataModalities object
    """
    def __init__(self, name):
        # Initialize variables
        self.name = name
        self.log_type = 'default'
        
    def add_modality(self, modality_name, modality_data):
        # Add data modality
        logit('Implement ADD_MODALITY function in DataModalities!', self.log_type)
        
    def save(self, filename):
        # Save
        logit('Implement SAVE function in DataModalities!', self.log_type)
        
    def load(self, filename):
        # Load
        logit('Implement LOAD function in DataModalities!', self.log_type)
        
    def set_log_type(self, log_type):
        # Change log type
        self.log_type = log_type
       