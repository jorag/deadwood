#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np
from mytools import *
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET # For reading .gpx files
from math import radians, cos, sin, asin, sqrt # for Haversine function - TODO, rewrite using numpy!


def pos2pix(geotransform, lat='default', lon='default', pixels_out = 'single', verbose=True):
    """Find pixel position by geotranform-info from GDAL.
    
    Input: geotranform, lat = , lon = , verbose = False 
    """
    func_log_id = "In pos2pix: "
    
    # TODO: ENSURE THAT INDICE RETURNED ARE POSITIVE
    # TODO: "lat" should be row and "lon" clolumn
    logit('WARNING! TEMPORARY FIX IN pos2pix TO ENSURE POSITIVE INDICE. LOOK INTO THIS!')
    
    # Get upper left corner and pixel width
    ul_lat = geotransform[3] # lat or y??
    ul_lon = geotransform[0] # long or x??
    pixel_width_lat = geotransform[5] # lat or y??
    pixel_width_lon = geotransform[1] # long or x??
    if verbose:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
                                  
    
    # Point of Interest
    point_lat = lat
    point_lon = lon
    if verbose:
        logit('\n lat =' + str(point_lat) + '\n lon =' + str(point_lon))
    
    # Estimate pixel position
    exact_pixpos_lat = (point_lat - ul_lat)/pixel_width_lat # Better way of ensuring correct sign? 
    exact_pixpos_lon = (ul_lon - point_lon)/pixel_width_lon # Better way of ensuring correct sign?
    
    # Return pixel position of closest the pixel or the four closest pixels  
    if pixels_out == 'single':
        pixpos_lat = int(round(exact_pixpos_lat))
        pixpos_lon = int(round(exact_pixpos_lon))
    elif pixels_out == 'npsingle':
        pixpos_lat = np.round(exact_pixpos_lat)
        pixpos_lat = pixpos_lat.astype(int)
        pixpos_lon = np.round(exact_pixpos_lon)
        pixpos_lon = pixpos_lon.astype(int)
    elif pixels_out == 'quad':
        # LAT: Check that the pixel indice are different
        lat_floor = int(np.floor(exact_pixpos_lat))
        lat_ceil = int(np.ceil(exact_pixpos_lat))
        if lat_floor != lat_ceil:
            pixpos_lat = [lat_floor, lat_ceil]
        else:
            pixpos_lat = lat_floor
            logit(func_log_id + 'pixels_out = ' + str(pixels_out) + ' found exact match (1 pixel) for lat = ' + str(lat))
            
        # LONG: Check that the pixel indice are different
        lon_floor = int(np.floor(exact_pixpos_lon))
        lon_ceil = int(np.ceil(exact_pixpos_lon))
        if lon_floor != lon_ceil:
            pixpos_lon = [lon_floor, lon_ceil]
        else:
            pixpos_lon = lon_floor
            logit(func_log_id + 'pixels_out = ' + str(pixels_out) + ' found exact match (1 pixel) for long = ' + str(lon))
        
    # Return pixel position
    return np.abs(pixpos_lat), np.abs(pixpos_lon)


def geocoords2pix(lat_band, lon_band, lat='default', lon='default', pixels_out = 'single', log_type=None):
    """Find pixel position by from latitude and longitude bands.
    
    Input: lat_band, lon_band, lat='default', lon='default', pixels_out = 'single', log_type=None
    """
    # TODO - move this elsewhere?
    coord_band = np.dstack((lat_band, lon_band))
    
    # Check input - TODO, check that lengths equal?
    if numel(lat) < 2 or numel(lon) < 2:
        lat = make_list(lat)
        lon = make_list(lon)
        
    # Initialize output lists
    pixpos_row = []
    pixpos_col = []

    # Loop over all input points 
    for i_point in range(length(lat)):
        # Initialize 3D point array
        coord_find = np.zeros((1,1,2))
        coord_find[0,0,0] = lat[i_point]
        coord_find[0,0,1] = lon[i_point]
    
        # Find distance between each coordinate in band (along third axis)
        # TODO - consider Haversine
        dists = np.linalg.norm(coord_band-coord_find, axis=2)
        # Find nearest coordinate (least distance)
        nearest = np.nanargmin(dists)
        #print(lat_band[np.unravel_index(nearest, lat_band.shape)], lon_band[np.unravel_index(nearest, lon_band.shape)])
        
        # Get back the original array indice
        if pixels_out.lower() in ['single', 'npsingle']:
            indice = np.unravel_index(nearest, lat_band.shape)
            pixpos_row.append(int(indice[0]))
            pixpos_col.append(int(indice[1]))
            lat_val = lat_band[int(indice[0]), int(indice[1])]
            lon_val = lon_band[int(indice[0]), int(indice[1])]
            if log_type is not None:
                logit('Coord found: '+str(lat_val)+'  '+str(lon_val)+'. Diff: '+str(lat_val-lat[i_point])+'  '+str(lon_val-lon[i_point]), log_type)
        else:
            raise NotImplementedError('pixels_out = ' + pixels_out + ' not implemented in geopixpos.geobands2pix()!')
    
    # Return pixel position
    if pixels_out.lower() in ['npsingle']:
        pixpos_row = np.asarray(pixpos_row)
        pixpos_col = np.asarray(pixpos_col)

    return pixpos_row, pixpos_col



def geobands2pix(lat_band, lon_band, lat='default', lon='default', pixels_out = 'single', verbose=False):
    """Find pixel position by from latitude and longitude bands.
    
    TODO: Remove this?
    Input: lat_band, lon_band, lat='default', lon='default', pixels_out = 'single', verbose=False
    """
    # Check input
    if numel(lat) < 2 or numel(lon) <2:
        lat = make_list(lat)
        lon = make_list(lon)
        
    # Initialize output lists
    pixpos_row = []
    pixpos_col = []
    # Loop over all input points 
    for i_point in range(length(lat)):
        # Current point of Interest
        point_lat = lat[i_point]
        point_lon = lon[i_point]
        
        # Could also look at smallest negative and smallest positive value??
        print((point_lat , point_lon))
        
        # Subtract position from band 
        lat_diff = np.abs(lat_band - point_lat)
        lon_diff = np.abs(lon_band - point_lon)
        #lat_diff = (lat_band - point_lat)
        #lon_diff = (lon_band - point_lon)
        
        # Find minimum (zero crossing) in each direction for each band
        lat_indice = np.where(lat_diff == np.min(lat_diff))
        lon_indice = np.where(lon_diff == np.min(lon_diff))
#        lat_indice = np.where(lat_diff[lat_diff>0] == np.min(lat_diff[lat_diff>0]))
#        lon_indice = np.where(lon_diff[lon_diff>0] == np.min(lon_diff[lon_diff>0]))
        
        
        # Convert to numpy arrays
        n_lat = length(lat_indice[0])
        n_lon = length(lon_indice[0])   
        print(n_lat ,'lat_indice:', lat_indice, '\n')
        print(n_lon ,'lon_indice:', lon_indice, '\n')
        # Use longest list of indice as search index
        if n_lat > n_lon:
            X = np.zeros((n_lat, 2))
            X[:,0] = lat_indice[0]
            X[:,1] = lat_indice[1]
            searched_values = np.zeros((n_lon, 2))
            searched_values[:,0] = lon_indice[0]
            searched_values[:,1] = lon_indice[1]
        else:
            X = np.zeros((n_lon, 2))
            X[:,0] = lon_indice[0]
            X[:,1] = lon_indice[1]
            searched_values = np.zeros((n_lat, 2))
            searched_values[:,0] = lat_indice[0]
            searched_values[:,1] = lat_indice[1]
                     
        # https://stackoverflow.com/questions/38674027/
        # Must be a match in BOTH row and column indice at the same time
        # Finds index in longest indice array where both indices match
        match = np.where((X==searched_values[:,None]).all(-1))[1]
        print('X ', X, '\n')
        print('searched_values ', searched_values, '\n')
        print(n_lat ,'lat_indice:', lat_indice, '\n')
        print(n_lon ,'lon_indice:', lon_indice, '\n')
        print(match)
        fig = plt.figure()
        plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
        # Plot data
        plt.scatter(X[:,0], X[:,1], c='b', marker = '+' )
        plt.scatter(searched_values[:,0], searched_values[:,1], c='r', marker = 'x')
        plt.xlabel("Row index")
        plt.ylabel("Column index")
        plt.legend(['Latitude', 'Longitude'], loc='lower right')
        
        # Check if there is an exact match, if not...
        if match.size == 0:
            # Set current minimum values to max (to find new minimum)
            lat_diff[np.where(lat_diff == np.min(lat_diff))] = np.max(lat_diff)
            lon_diff[np.where(lon_diff == np.min(lon_diff))] = np.max(lon_diff)
            
            lat_indice2 = np.where(lat_diff == np.min(lat_diff)) 
            lon_indice2 = np.where(lon_diff == np.min(lon_diff)) 
            X2 = np.zeros((length(lat_indice2[0]), 2))
            X2[:,0] = lat_indice2[0]
            X2[:,1] = lat_indice2[1]
            X2 = np.vstack((X, X2))
            
            X4 = np.zeros((length(lon_indice2[0]), 2))
            X4[:,0] = lon_indice2[0]
            X4[:,1] = lon_indice2[1]
            X4 = np.vstack((searched_values, X4))
            # Plot data
            plt.scatter(X2[0], X2[1], c='b', marker = '+' )
            plt.scatter(X4[0], X4[1], c='r', marker = 'x')
            plt.xlabel("Row index")
            plt.ylabel("Column index")
            plt.legend(['Latitude', 'Longitude'], loc='lower right')
            print(match)
        
        # Get back the original array indice
        if pixels_out.lower() in ['single', 'npsingle']:
            i_match = 0 # TODO: Change how multiple matches are handled??
            pixpos_row.append(int(X[match[i_match] , 0]))
            pixpos_col.append(int(X[match[i_match] , 1]))
            lat_val = lat_band[X[match[i_match],0], X[match[i_match],1]]
            lon_val = lon_band[X[match[i_match],0], X[match[i_match],1]]
            print((lat_val, lon_val))
        else:
            raise NotImplementedError('pixels_out = ' + pixels_out + ' not implemented in geopixpos.geobands2pix()!')
    
    # Return pixel position
    if pixels_out.lower() in ['npsingle']:
        pixpos_row = np.asarray(pixpos_row)
        pixpos_col = np.asarray(pixpos_col)

    return pixpos_row, pixpos_col


def refinepos(a):
    """TODO: Implement this!
    
    Idea: Use the Geo-transform info to get a rough estimate of position, then look up "exact" position using LAT/LONG bands
    Look at sign of band minus coordinate of interest
    Use mean(abs(diff)) as sanity check, shuld be much smaller than 1 if correct (lat/long) band
    
    Needs GDAL (import). 
    Needs indice of lat/long bands.
    """
    lat_band = dataset.GetRasterBand(lat_band_i)
    print("Band Type={}".format(gdal.GetDataTypeName(lat_band.DataType)))
    
    # Offset in pixels from estimated position so that the estimated position is at the centre
    # Read this many extra pixels in each direction
    pix_offset_x = 2
    pix_offset_y = 2
    
    # Read geoposition band for fine search of position
    data = lat_band.ReadAsArray(pixpos_lon - pix_offset_x, pixpos_lat - pix_offset_y, pix_offset_x*2+1, pix_offset_y*2+1)
    #data = lat_band.ReadAsArray(int(exact_pixpos_lon), int(exact_pixpos_lat), 3, 3)
    print(data)
    
    diff_data = data-point_lat
    abs_diff = abs(diff_data)
    print(diff_data)
    a = np.where(abs_diff == np.min(abs_diff))
    b = np.argmin(abs_diff)
    
    # In the original example, there point of interests latitude is equal distance between 69.99997 lat and 70.00003 lat
    # In this case, there should be no refinement to the original estimate
    # Or should perhaps both be considered?
    
    print(data[a]) 
    # Does not work for b - this is because of how argmin works in Python:
    # In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
    
    
def gdalinfo_log(dataset, log_type='default'):
    """Log dataset info from GDAL.
    
    Input: dataset, log_type='default'
    """
    # Print information - can also use command line: !gdalinfo file_path 
    logit("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName), log_type)
    logit("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount), log_type)
    logit("Projection is {}".format(dataset.GetProjection()), log_type)
    
    
def bandinfo_log(band, log_type='default'):
    """Log band info from GDAL.
    
    Input: band, log_type='default'
    """
    # To avoid importing GDAL in this module...
    logit("Band Type={}".format(band.DataType), log_type)
    #logit("Band Type={}".format(gdal.GetDataTypeName(band.DataType)), log_type)

    # Get pixel info
    pixel_min = band.GetMinimum()
    pixel_max = band.GetMaximum()
    if not pixel_min or not pixel_max:
        (pixel_min,pixel_max) = band.ComputeRasterMinMax(True)
    logit("Min={:.3f}, Max={:.3f}".format(pixel_min, pixel_max), log_type)
          
    if band.GetOverviewCount() > 0:
        logit("Band, number of overviews:", log_type)
        logit(band.GetOverviewCount(), log_type)
          
    if band.GetRasterColorTable():
        logit("Band, number of colour table with entries:", log_type)
        logit(band.GetRasterColorTable().GetCount(), log_type)
        
def arrayinfo_log(array, log_type='default'):
    """Log band info from array.
    
    Input: array, log_type='default'
    """

    logit('Band min:'+ str(np.min(array[array>0])) + ', Band max:'+ str(np.max(array[array>0])), log_type='default')
    logit('Array contains NaN? -  ' + str(np.isnan(array).any()), log_type='default')
        
        
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    From: https://stackoverflow.com/questions/4913349/
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def gpxgeobox(gpx_in, lat_band, lon_band, margin=(0,0), log_type='default'):
    """Get indices for GeoTIFF product bounding box from .gpx file bounds.
    
    Input: gpx_in, lat_band, lon_band
    """
    # Check optional margin argument?
    
    # Load .gpx data
    xml_tree = ET.parse(gpx_in)
    # Get <metadata><bounds> element
    bounds = xml_tree.findall('//{http://www.topografix.com/GPX/1/1}bounds')[0]
    # Read bounds
    lat_bounds = [bounds.attrib['minlat'], bounds.attrib['maxlat']]
    lon_bounds = [bounds.attrib['minlon'], bounds.attrib['maxlon']]
        
    # Go through all combinations and find row and column indices
    row_ind = []
    col_ind = []
    for lat in lat_bounds:
        for lon in lon_bounds:
            row, col = geocoords2pix(lat_band, lon_band, lat=lat, lon=lon, pixels_out = 'npsingle')
            row_ind.append(row)
            col_ind.append(col)
            

    # Find indices for rectangle, add margin and ensure it is within the range 
    r_min = max(np.min(row_ind) - margin[0], 0)
    r_max = min(np.max(row_ind) + margin[0], lat_band.shape[0])
    c_min = max(np.min(col_ind) - margin[1], 0)
    c_max = min(np.max(col_ind) + margin[1], lat_band.shape[1])

    return (r_min,r_max), (c_min,c_max)

    