#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:17:30 2018

@author: jorag
@author of length: MadsAdrian, updated by jorag 
"""

import numpy as np

def length(x):
    """Returns length of input.
    
    Mimics MATLAB's length function.
    """
    if isinstance(x, (int, float, complex)):
        return 1
    elif isinstance(x, np.ndarray):
        return max(x.shape)
    try:
        return len(x)
    except TypeError as e:
        print('In length, add handling for type ' + str(type(x)))
        raise e
        
def numel(x):
    """Returns the number of elements in the input.
    
    Mimics MATLAB's numel function.
    """
    if isinstance(x, str):
        return 1
    elif isinstance(x, tuple):
        return 1
    else:
        return length(x)
    
def make_list(x):
    """Make the input into a list.
    
    Currently only for inputs with one element.
    """
    if numel(x) == 1:
        if isinstance(x, (int, float, complex, str)):
            return [x]
        elif isinstance(x, np.ndarray):
            raise NotImplementedError('Implement support for numpy arrays in mytools.makelist(x)!')
        elif isinstance(x, tuple):
            # TODO: Should each tuple element be a list element?
            return [x]
        else:
            raise NotImplementedError('Implement support for data type in mytools.makelist(x)!')
    else:
        raise NotImplementedError('Implement support for data with numel(x) > 1 in mytools.makelist(x)!')

        
        
def mynormal(x, params):
    """Univariate normal probability density function.
    
    Input:
        x - point(s)
        params - contains mu and sigma (in that order)
    Output:
        pdf(x)
    """
    mu = params[0]
    sigma = params[1]
    
    pdf = 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-(x - mu)**2/(2*sigma**2))
    return pdf


def logit(txt_in, log_type='print'):
    """Function for logging text to file or printing to console.
    
    Enables a unified command for printing text during development/debugging
    and logging the text to a file during normal (mature) operations.
    """
    if log_type == 'print':
        print(txt_in)
    elif log_type == 'default':
        # TODO: Implement this (read default state)
        print(txt_in)
        
    return
