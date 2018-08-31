#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np

def pos2pix(geotransform, **keywords):
    """Docstring for function.
    
    Fill in.
    """
    ul_lat = geotransform[3] # lat or y??
    ul_lon = geotransform[0] # long or x??
    pixel_width_lat = geotransform[5] # lat or y??
    pixel_width_lon = geotransform[1] # long or x??
    
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
                                  
    
    # Point of Interest
    point_lat = 70.0
    point_lon = 27.0

    # Estimate pixel position
    exact_pixpos_lat = (point_lat - ul_lat)/pixel_width_lat # Better way of ensuring correct sign? 
    exact_pixpos_lon = (ul_lon - point_lon)/pixel_width_lon # Better way of ensuring correct sign?
    pixpos_lat = int(round(exact_pixpos_lat))
    pixpos_lon = int(round(exact_pixpos_lon))
    
    # Try to use keywords for arguments, e.g. lat=70.123
    for kw in keywords:
        print(kw, ":", keywords[kw])
        
    # Return pixel position
    return pixpos_lat, pixpos_lon
