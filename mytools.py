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


def norm01(input_array, norm_type='global', min_cap=None, max_cap=None, min_cap_value=np.NaN, max_cap_value=np.NaN, log_type=None):
    """Normalize data.
    
    Parameters:
    norm_type:
        'none' - return input array (might print min, max and mean)
        'local' - min and max of each variable (column) is 0 and 1
        'global' - min of array is 0, max is 1
    min_cap: Truncate values below this value to min_cap_value before normalizing
    max_cap: Truncate values above this value to max_cap_value before normalizing
    """
    
    # Replace values outside envolope/cap with NaNs (or specifie value)
    if min_cap is not None:
        input_array[input_array  < min_cap] = min_cap_value
                   
    if max_cap is not None:
        input_array[input_array > max_cap] = max_cap_value
    
    # Normalize data for selected normalization option
    if norm_type.lower() in ['none']:
        # No normalization (do not return yet, might print min, max, and mean)
        output_array = input_array
    elif norm_type.lower() in ['global', 'all', 'set']:
        # Normalize to 0-1 (globally)
        output_array = input_array - np.nanmin(input_array)
        output_array = output_array/np.nanmax(output_array)
    elif norm_type.lower() in ['local', 'separate']:
        # Normalize to 0-1 for each column
        output_array = input_array - np.nanmin(input_array, axis=0)
        output_array = output_array/np.nanmax(output_array, axis=0)
    elif norm_type.lower() in ['band', 'channel']:
        # Normalize to 0-1 for each channel (assumed to be last index)
        # Get shape of input
        input_shape = input_array.shape
        output_array = np.zeros(input_shape)
        # Normalize each channel
        for i_channel in range(0, input_shape[2]):
            output_array[:,:,i_channel] = input_array[:,:,i_channel] - np.nanmin(input_array[:,:,i_channel])
            output_array[:,:,i_channel] = output_array[:,:,i_channel]/np.nanmax(output_array[:,:,i_channel])
    
    # Log / print results
    if log_type is not None:
        logit('\n INPUT:', log_type)
        logit(np.nanmean(input_array), log_type)
        logit(np.nanmax(input_array, axis=0), log_type)
        logit(np.nanmean(input_array, axis=0), log_type)
        logit(np.nanmin(input_array, axis=0), log_type)
        
        logit('\n OUTPUT:', log_type)
        logit(np.nanmean(output_array), log_type)
        logit(np.nanmax(output_array, axis=0), log_type)
        logit(np.nanmean(output_array, axis=0), log_type)
        logit(np.nanmin(output_array, axis=0), log_type)
        
    return output_array


def mycolourvec(shuffle=False):
    """Get a standardized colour vector (list) for plotting.
    
    Shorthand to avoid defining it for every plot that needs different colours 
    to distinguish between classes etc.
    """
    # Define vector with colours
    colour_vec = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    
    # Randomize order?
    if shuffle:
        np.random.shuffle(colour_vec)
        
    return colour_vec


def dB(x, ref=1, input_type='power'):
    """Return result, x, in decibels (dB) relative to reference, ref."""    
    if input_type.lower() in ['power', 'pwr']:
        a = 10
    elif input_type.lower() in ['aplitude', 'amp']:
        a = 20
    return a*(np.log10(x) - np.log10(ref))


def identity(x, **kwargs):
    """Identity function that ignores extra arguments."""    
    return x 


def pauli_rgb(x):
    """Create the Pauli RGB image from input.
    
    If the number of channels in is 3 or 4, the input is assumed to be complex.
    If the number of channels in is 6 or 8, the input is assumed to be I and Q.
    (Inphase and Quadrature components in sequential band order.)
    Either way, the values are assumed to represent amplitude (not intensity).
    """
    shape_in = x.shape
    # Initialize output
    rgb_out = np.zeros((shape_in[0], shape_in[1], 3))
    # Number of bands determines form of expression
    if shape_in[2] == 4:
        # Form is complex arrays: HH, HV, VH, VV
        rgb_out[:,:,1] = 0.5*np.abs(x[:,:,1] + x[:,:,2])**2 # G
        rgb_out[:,:,0] = 0.5*np.abs(x[:,:,0] - x[:,:,3])**2 # R
        rgb_out[:,:,2] = 0.5*np.abs(x[:,:,0] + x[:,:,3])**2 # B
    elif shape_in[2] == 3:
        # Form is complex arrays: HH, HV, VV  (reciprocity assumed) 
        rgb_out[:,:,1] = 0.5*np.abs(x[:,:,1])**2 # G
        rgb_out[:,:,0] = 0.5*np.abs(x[:,:,0] - x[:,:,2])**2 # R
        rgb_out[:,:,2] = 0.5*np.abs(x[:,:,0] + x[:,:,2])**2 # B
    elif shape_in[2] == 8:
        # Form is real arrays: i_HH, q_HH, i_HV, q_HV, i_VH, q_VH, i_VV, q_VV
        rgb_out[:,:,1] = 0.5*((x[:,:,2]+x[:,:,4])**2 + (x[:,:,3]+x[:,:,5])**2) # G
        rgb_out[:,:,0] = 0.5*((x[:,:,0]-x[:,:,6])**2 + (x[:,:,1]-x[:,:,7])**2) # R
        rgb_out[:,:,2] = 0.5*((x[:,:,0]+x[:,:,6])**2 + (x[:,:,1]+x[:,:,7])**2) # B
    elif shape_in[2] == 6:
        # Form is real arrays: i_HH, q_HH, i_HV, q_HV, i_VV, q_VV (reciprocity assumed) 
        rgb_out[:,:,1] = 0.5*((x[:,:,2])**2 + (x[:,:,3])**2) # G
        rgb_out[:,:,0] = 0.5*((x[:,:,0]-x[:,:,4])**2 + (x[:,:,1]-x[:,:,5])**2) # R
        rgb_out[:,:,2] = 0.5*((x[:,:,0]+x[:,:,4])**2 + (x[:,:,1]+x[:,:,5])**2) # B
               
    return rgb_out 


def iq2complex(x, reciprocity=False):
    """Merge I and Q bands to complex valued array.
    
    Create an array with complex values from separate, real-vauled Inphase and 
    Quadrature components.
    """
    shape_in = x.shape
    # Number of bands determines form of expression
    if reciprocity and shape_in[2] == 8:
        # Initialize output
        array_out = np.zeros((shape_in[0], shape_in[1], 3), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VH, q_VH, i_VV, q_VV
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = (x[:,:,2] + 1j*x[:,:,3] + x[:,:,4] + 1j*x[:,:,5])/2 # HV (=VH)
        array_out[:,:,2] = x[:,:,6] + 1j * x[:,:,7] # VV
    elif not reciprocity and shape_in[2] == 8:
        # Initialize output
        array_out = np.zeros((shape_in[0], shape_in[1], 4), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VH, q_VH, i_VV, q_VV
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = x[:,:,2] + 1j * x[:,:,3] # HV
        array_out[:,:,2] = x[:,:,4] + 1j * x[:,:,5] # VH
        array_out[:,:,3] = x[:,:,6] + 1j * x[:,:,7] # VV
    elif shape_in[2] == 6:
        # Initialize output
        array_out = np.zeros((shape_in[0], shape_in[1], 3), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VV, q_VV (reciprocity assumed) 
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = x[:,:,2] + 1j * x[:,:,3] # HV (=VH)
        array_out[:,:,2] = x[:,:,4] + 1j * x[:,:,5] # VH
               
    return array_out 
