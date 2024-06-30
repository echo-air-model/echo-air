#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Data Object

@author: libbykoolik
last modified: 2024-06-13
"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import matplotlib.pyplot as plt
import pyarrow
from scipy.io import netcdf_file as nf
import os
from os import path
import sys
from inspect import currentframe, getframeinfo
sys.path.append('./scripts')
from tool_utils import *

#%% Define the Population Object
class population:
    '''
    Defines a new object for storing and manipulating concentration data.
    
    INPUTS:
        - file_path: the file path of the raw population data
        - isrm_obj: the ISRM object, as defined by isrm.py
        - load_file: a Boolean indicating whether or not the file should be loaded 
        - verbose: a Boolean indicating whether or not detailed logging statements 
          should be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
          
    CALCULATES:
        - valid_file: a Boolean indicating whether or not the file provided is valid
        - geometry: geospatial information associated with the emissions input
        - pop_all: complete, detailed population data from the source
        - pop_geo: a geodataframe with population IDs and spatial information
        - crs: the inherent coordinate reference system associated with the emissions input
        - pop_exp: a geodataframe containing the population information with associated 
          spatial information, summarized across age bins
        - pop_hia: a geodataframe containing the population information with associated
          spatial information, broken out by age bin
          
    EXTERNAL FUNCTIONS:
        - allocate_population: reallocates population into new geometry using a 
          spatial intersect
        
    '''
    def __init__(self, file_path, debug_mode, load_file=True, verbose=False):
        ''' Initializes the Population object'''        
        
        # Gather meta data
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1].lower()
        self.load_file = load_file
        self.verbose = verbose
        self.debug_mode = debug_mode
        
        # Return a starting statement
        verboseprint(self.verbose, '- [POPULATION] Creating a new population object from {}'.format(self.file_path),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Initialize population object by reading in the feather file
        self.valid_file = self.check_path()
        
        if not self.valid_file:
            logging.info('\n << [POPULATION] ERROR: The filepath provided is not correct. Please correct and retry. >>')
            sys.exit()
        
        # Read in the data
        if self.load_file == True and self.valid_file:
            verboseprint(self.verbose, '- [POPULATION] Attempting to load the population data. This step may take some time.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.pop_all, self.pop_geo, self.crs = self.load_population()
            self.pop_exp = self.make_pop_exp()
            self.pop_hia = self.make_pop_hia()
            verboseprint(self.verbose, '- [POPULATION] Population data successfully loaded.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))        
            
    def __str__(self):
        return '< Population object for year '+str(self.year)+ '>'

    def __repr__(self):
        return '< Emissions object created from '+self.file_path + '>'

    def check_path(self):
        ''' Checks if file exists at the path specified '''
        # Use the os library to check the path and the file
        path_exists = path.exists(self.file_path)
        file_exists = path.isfile(self.file_path)
        return path_exists and file_exists
    
    def load_population(self):
        ''' Loads the population file, depending on the extension ''' 
        # Based on the file extension, run different load functions
        if self.file_type == 'shp':
            pop_all = self.load_shp()
        
        if self.file_type == 'feather':
            pop_all = self.load_feather()
            
        # Create a variable that is just geometry and IDs
        pop_geo = pop_all[['POP_ID','geometry']].copy().drop_duplicates()
        pop_crs = pop_geo.crs
        
        return pop_all, pop_geo, pop_crs
    
    def load_shp(self):
        ''' Loads population data from a shapefile. '''
        # Shapefiles are read using geopandas
        pop_all = gpd.read_file(self.file_path)
        
        return pop_all
                
    def load_feather(self):
        ''' Loads population data from a feather file. '''
        # Feather file is read using geopandas
        pop_all = gpd.read_feather(self.file_path)
        
        return pop_all

    def make_pop_exp(self):
        ''' Creates the population exposure object '''
        # Create a copy of the population data to avoid overwriting
        pop_tmp = self.pop_all.copy()
        
        ## Create the exposure calculation population object
        # For the exposure calculations, we do not need the age bins
        pop_exp = pop_tmp[['POP_ID', 'YEAR', 'TOTAL', 'ASIAN', 'BLACK', 'HISLA', 
                           'INDIG', 'PACIS', 'WHITE', 'OTHER']].copy()
        
        # Sum across POP_ID and YEAR
        pop_exp = pop_exp.groupby(['POP_ID','YEAR'])[['TOTAL', 'ASIAN', 'BLACK', 
                                                      'HISLA', 'INDIG', 'PACIS', 
                                                      'WHITE', 'OTHER']].sum().reset_index()
        
        # Add geometry back in
        pop_exp = pd.merge(self.pop_geo, pop_exp, on='POP_ID')
        
        return pop_exp
    
    def make_pop_hia(self):
        ''' Creates the population exposure object for hia calculations '''
        # Creates a copy of the population data to avoid overwriting
        pop_hia = self.pop_all.copy()
        
        # Simple update
        pop_hia['START_AGE'] = pop_hia['START_AGE'].astype(int)
        pop_hia['END_AGE'] = pop_hia['END_AGE'].astype(int)
        
        return pop_hia

    def project_pop(self, pop_obj, new_crs):
        ''' Projects the population data into a new crs '''
        pop_obj_prj = pop_obj.to_crs(new_crs)
    
        return pop_obj_prj

    def create_crosswalk(self, pop_obj, new_geometry, new_geometry_ID):
        ''' Reallocates the population into the new geometry using a spatial intersect '''
        # Print to the log file
        verboseprint(self.verbose, '- [POPULATION] Calculating crosswalk between population geography and ISRM geography', 
            self.debug_mode, frameinfo=getframeinfo(currentframe()))

        # Define the racial/ethnic groups and estimate the intersection population
        cols = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE','OTHER']
        
        # Find old total
        old_pop_total = pop_obj[cols].sum()

        # Create a deep copy of pop_geo that is just the unique geographies and associated POP_ID (should be n = 8057)
        pop_geo = pop_obj[['POP_ID', 'geometry']].drop_duplicates().copy(deep=True)

        # Perform the intersection between unique_pop_geo and new_geometry
        intersect = gpd.overlay(pop_geo, new_geometry, how='intersection')

        # Get total area of each input geometry cell
        intersect['AREA_M2'] = intersect.geometry.area / (1000 * 1000)
        total_area = intersect.groupby('POP_ID').sum()['AREA_M2'].to_dict()

        # Add a total area and area fraction to the intersect object
        intersect['total_area'] = intersect['POP_ID'].map(total_area)
        intersect['area_frac'] = intersect['AREA_M2'] / intersect['total_area']

        # Create the crosswalk with columns: POP_ID, new_geometry_ID, area_frac
        crosswalk = intersect[['POP_ID', new_geometry_ID, 'area_frac']].copy()

        return crosswalk, old_pop_totals
    
    def allocate_population(self, new_geometry, age_stratified=False):
        """
        Allocates population from population input file to ISRM grid cells using the crosswalk.

        Parameters:
        - age_stratified: Boolean indicating if the population data is age-stratified.
        - new_geometry: isrm geodata

        Returns:
        - new_pop: DataFrame with allocated population data by ISRM_ID (and age groups if age_stratified).
        """
        # Logging print statements
        if age_stratified:
            verboseprint(self.verbose, '- [HEALTH] Allocating age-stratified population from population input file to ISRM grid cells.',
                        self.debug_mode, frameinfo=getframeinfo(currentframe()))
            pop_data = self.pop_hia
        else:
            verboseprint(self.verbose, '- [POPULATION] Allocating total population from population input file to ISRM grid cells.',
                        self.debug_mode, frameinfo=getframeinfo(currentframe()))
            pop_data = self.pop_exp

        # Create a crosswalk and old totals object
        self.crosswalk, self.old_totals = self.create_crosswalk(self.pop_exp, geodata, 'ISRM_DATA') 

        # Merge population data with the crosswalk on POP_ID
        merged_pop = pop_data.merge(self.crosswalk, on='POP_ID')

        # Multiply each group's population count by the area fraction
        cols = ['TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']
        for c in cols:
            merged_pop[c] = merged_pop[c] * merged_pop['area_frac']

        # Sum the populations by ISRM_ID (and age groups if age_stratified)
        if age_stratified:
            gb_cols = ['ISRM_ID', 'START_AGE', 'END_AGE']
        else:
            gb_cols = ['ISRM_ID']

        new_pop = merged_pop.groupby(gb_cols)[cols].sum().reset_index()

        # Confirm new total populations match
        new_pop_total = new_pop[cols].sum()

        for c in cols:
            assert np.isclose(self.old_totals[c], new_pop_total[c]), f"Population mismatch in column {c}"

        # Print confirmation
        if age_stratified:
            verboseprint(self.verbose, '- [HEALTH] Census tract population data successfully re-allocated to the ISRM grid.',
                        self.debug_mode, frameinfo=getframeinfo(currentframe()))
        else:
            verboseprint(self.verbose, '- [POPULATION] Census tract population data successfully re-allocated to the ISRM grid.',
                        self.debug_mode, frameinfo=getframeinfo(currentframe()))

        return new_pop
