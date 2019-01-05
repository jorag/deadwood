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


def get_label_weights(labels):
    """Create a vector with weights for each label according to number of 
    occurances of each label. 
    
    Moved from dataclass.py
    """
    # Find the unique labels and counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # Number of points
    n_points = np.sum(label_counts)
    # Set the default weights
    weights = np.ones((length(labels),)) / n_points
    # Get relative weights of each class
    #rel_weights = label_counts/n_points
    for i_label in unique_labels:
        weights[np.where(labels==i_label)] = label_counts[i_label]/n_points 
                
    # Normalize weights and return
    weights = weights/np.sum(weights)
    return weights

        
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


def imtensor2array(input_im):
    """Convert 3D image array on the form channel, x, y to 2D array with 
    channels as columns.
    
    Useful for classification or normalization.
    """
    # Get number of channels etc.
    n_channels = input_im.shape[0]
    n_rows = input_im.shape[1]
    n_cols = input_im.shape[2]
    # Reshape array to n_cols*n_rows rows with the channels as columns 
    input_im = np.transpose(input_im, (1, 2, 0)) # Change order to rows, cols, channels
    output_im = np.reshape(input_im, (n_rows*n_cols, n_channels))
        
    return output_im, n_rows, n_cols 


def norm01(input_array, min_cap = None, max_cap = None, log_type='print'):
    """Normalize.
    
    Use for normalization
    """
    
    # Replace values outside envolope/cap with NaNs
    if min_cap is not None:
        input_array[input_array  < min_cap] = np.NaN
                   
    if max_cap is not None:
        input_array[input_array > max_cap] = np.NaN
    
    valid_min = np.min(input_array[input_array>-9999])

    print(np.nanmax(input_array))
    print(np.nanmin(input_array))
    print(np.nanmax(input_array, axis=0))
    print(np.nanmin(input_array, axis=0))
        
    return 
