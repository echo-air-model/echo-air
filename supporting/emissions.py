#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emissions Data Object

@author: libbykoolik
last modified: 2024-02-15
"""

# Import Libraries
import sys
import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from inspect import currentframe, getframeinfo
sys.path.append('./scripts')
from tool_utils import *


#%% Define the Emissions Object
class emissions:
    '''
    Defines a new object for storing and manipulating emissions data.
    
    INPUTS:
        - file_path: File path to an emissions file (allows .shp, .feather, and .csv only)
        - output_dir: a filepath string for the output directory
        - f_out: a string containing the filename pattern to be used in output files
        - units: Units of the emissions file (default is ug/s)
        - name: plain English tag for emissions, otherwise will use the filename
        - details_to_keep: any additional aggregation field (e.g., FUEL_TYPE)
        - load_file: set to True to import emissions, otherwise will just run checks
        - verbose: enable for more detailed outputs
        - debug_mode: a Boolean indicating whether or not to output debug statements
        
    CALCULATES:
        - PM25: primary PM2.5 emissions in each grid cell
        - NH3: ammonia emissions in each grid cell
        - VOC: VOC compound emissions in each grid cell
        - NOX: NOx emissions in each grid cell
        - SOX: SOx emissions in each grid cell
        - L0_flag, L1_flag, L2_flag, isrm_hole_flag: Booleans indicating whether 
          each layer should be calculated based on emissions release heights
          
    EXTERNAL FUNCTIONS:
        - get_pollutant_layer: pulls a single pollutant layer based on pol_name
        - visualize_emissions: creates a simple map of emissions for a provided 
          pollutant

    '''
    def __init__(self, file_path, output_dir, f_out, debug_mode, units='ug/s', name='', details_to_keep=[], filter_dict={}, load_file=True, verbose=False):
        ''' Initializes the emissions object'''     
        
        # Initialize path and check that it is valid
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1].lower()
        self.valid_file = self.check_path()

        # Define additional Meta Variables
        self.emissions_name = self.get_name(name)
        self.details_to_keep = details_to_keep
        self.filter_dict = filter_dict
        self.filter = bool(self.filter_dict) # returns False if empty, True if not empty
        self.debug_mode = debug_mode
        self.verbose = verbose
        self.output_dir = output_dir
        self.f_out = f_out

        # Return a starting statement
        verboseprint(self.verbose, '- [EMISSIONS] Creating a new emissions object from {}'.format(self.file_path),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # If the file does not exist, quit before opening
        if not self.valid_file:
            logging.info('\n<< [EMISSIONS] ERROR: The emissions filepath provided is not correct. Please correct and retry. >>')
            sys.exit()
        else:
            verboseprint(self.verbose, '- [EMISSIONS] Filepath and file found. Proceeding to import emissions data.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Confirm Units are Valid
        self.units = units 
        self.valid_units = self.check_units()
        if not self.valid_units:
            logging.info('\n<< [EMISSIONS] ERROR: The units provided ({}) are not valid. Please convert emissions to ug/s (micrograms per second) and try again. >>'.format(self.units))
            sys.exit()
        else:
            verboseprint(self.verbose, '- [EMISSIONS] Units of provided emissions ({}) are valid.'.format(self.units),
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            
        # If load_file is True, import the emissions data
        if load_file == True and self.valid_file:
            verboseprint(self.verbose, '- [EMISSIONS] Attempting to load the emissions data. This step may take some time.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.geometry, self.emissions_data, self.crs = self.load_emissions()            
            verboseprint(self.verbose, '- [EMISSIONS] Emissions successfully loaded.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
            self.emissions_data.columns = map(str.upper, self.emissions_data.columns)
                        
            # Check the emissions data
            self.valid_emissions = self.check_emissions()
            
            # Check for the 'HEIGHT_M' column; if it exists, add to details_to_keep, otherwise add HEIGHT_M = 0 column
            self.check_height()
            
            # If the emissions are valid, continue
            if self.valid_emissions:
                # Update to convert everything to polygons
                self.check_geo_types()

                # Checking max area
                self.check_polygon_area()
            
                # Print statements
                verboseprint(self.verbose, '- [EMISSIONS] Emissions formatting has been checked and confirmed to be valid.',
                             self.debug_mode, frameinfo=getframeinfo(currentframe()))
                verboseprint(self.verbose, '- [EMISSIONS] Beginning data cleaning and processing.',
                             self.debug_mode, frameinfo=getframeinfo(currentframe()))
                
                # Should add a statement about details_to_keep and the aggfunc
                self.emissions_data_clean = self.clean_up(self.details_to_keep)
                self.PM25 = self.split_pollutants(self.emissions_data_clean, 'PM25', self.details_to_keep)
                self.NH3 = self.split_pollutants(self.emissions_data_clean, 'NH3', self.details_to_keep)
                self.NOX = self.split_pollutants(self.emissions_data_clean, 'NOX', self.details_to_keep)
                self.VOC = self.split_pollutants(self.emissions_data_clean, 'VOC', self.details_to_keep)
                self.SOX = self.split_pollutants(self.emissions_data_clean, 'SOX', self.details_to_keep)
                
                # Which ISRM layers are needed?
                self.L0_flag, self.L1_flag, self.L2_flag, self.isrm_hole_flag = self.which_layers()
    
    def __str__(self):
        return 'Emissions object created from '+self.file_path

    def __repr__(self):
        return '< Emissions object created from '+self.file_path+'>'


    def get_file_path(self):
        ''' Returns the file path '''
        print('Emissions file path: ', self.file_path)
        return self.file_path
    
    def get_name(self, name):
        ''' Gets emissions file name from the file_path unless provided '''
        if name == '':
            emissions_name = path.basename(self.file_path).split('.')[0]
        else:
            emissions_name = name
        return emissions_name
    
    def get_unit_conversions(self):
        ''' Hard-coded dictionary of unit conversions '''
        # The purpose of this function is to create a singular place where units have
        # to be updated if future iterations add more units.
        
        # mass_units and time_units are dictionaries with keys = current units and values = conversion factor to ug or s
        mass_units = {'ug':1.0, 'g':10.0**6, 'lb':453592370.0, 'ton':907184740000.0, 
                      'mt':1000000000000.0, 'kg':10.0**9} # micrograms per unit
        
        time_units = {'s': 1.0, 'min':60.0, 'hr':3600.0, 'day':86400.0, 'yr':31536000.0}
        
        return mass_units, time_units
    
    def check_path(self):
        ''' Checks if file exists at the path specified '''
        # Use the os library to check the path and the file
        path_exists = path.exists(self.file_path)
        file_exists = path.isfile(self.file_path)
        return path_exists and file_exists

    def check_units(self):
        ''' Checks if units are compatible with program '''
        units = self.units.lower() # force lower for simplicity
        
        # Make this somewhat flexible for more possibilities
        unit_mass, unit_time = units.split('/')
        mass_units, time_units = self.get_unit_conversions()
        
        # Check if units are in the dictionaries
        valid_units = (unit_mass in mass_units.keys()) and (unit_time in time_units.keys()) 
        
        return valid_units
    
    def load_emissions(self):
        ''' Loads the emissions file, depending on the extension ''' 
        # Based on the file extension, run different load functions
        if self.file_type == 'shp':
            geometry, emissions_data, crs = self.load_shp()
        
        elif self.file_type == 'feather':
            geometry, emissions_data, crs = self.load_feather()
        
        elif self.file_type == 'csv':
            geometry, emissions_data, crs = self.load_csv()
            
        else:
            raise ValueError('Emissions file is of an unknown type. Cannot proceed')
        
        # Project to the desired CRS if not already in that CRS
        desired_crs = 'epsg:3310'  # California NAD83 Albers (m)
        if crs != desired_crs:
            geometry = geometry.to_crs(desired_crs)

        return geometry, emissions_data, crs
    
    def check_id(self, emissions_gdf):
        ''' Checks to see if I_CELL and J_CELL are included in emissions file. If missing, adds them. '''
        
        # Look for I_CELL
        has_i_cell = 'I_CELL' in emissions_gdf.columns
        
        # Look for J_CELL
        has_j_cell = 'J_CELL' in emissions_gdf.columns
        
        # If either is missing, return an error and quit
        if not (has_i_cell and has_j_cell):
            
            # Return an error
            logging.info('\n<< [EMISSIONS] WARNING: The emissions input is missing the I_CELL and/or J_CELL column. The tool will add these. Please include identifiers in the future.')
            
            # Add unique identifiers for I_CELL and J_CELL
            if not has_i_cell:
                emissions_gdf['I_CELL'] = np.arange(emissions_gdf.shape[0])

            if not has_j_cell:
                emissions_gdf['J_CELL'] = np.arange(emissions_gdf.shape[0])
        
        return emissions_gdf
                
    def load_shp(self):
        ''' 
        Loads emissions data from a shapefile.
        
        Requirements:
            - Emissions data must have the columns I_CELL and J_CELL that should be uniquely 
            indexed to spatial information
        
        '''
        # Add log statements
        verboseprint(self.verbose, '- [EMISSIONS] Reading emissions from a shapefile.',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Shapefiles are read using geopandas
        emissions_gdf = gpd.read_file(self.file_path)
        
        # Check for I_CELL/J_CELL
        emissions_gdf = self.check_id(emissions_gdf)
        
        # Split off geometry from emissions data
        geometry = emissions_gdf[['I_CELL','J_CELL','geometry']]\
            .copy().drop_duplicates().reset_index(drop=True)
        emissions_data = pd.DataFrame(emissions_gdf.drop(columns='geometry'))
        
        # Separately save the coordinate reference system
        crs = emissions_gdf.crs
        
        return geometry, emissions_data, crs
    
    def load_feather(self):
        ''' 
        Loads emissions data from a feather.
        
        Requirements:
            - Emissions data must have the columns I_CELL and J_CELL that should be uniquely 
            indexed to spatial information
        
        '''
        # Add log statements
        verboseprint(self.verbose, '- [EMISSIONS] Reading emissions from a feather file.',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Feathers are read using geopandas
        emissions_gdf = gpd.read_feather(self.file_path)
        
        # Check for I_CELL/J_CELL
        emissions_gdf = self.check_id(emissions_gdf)
        
        # Split off geometry from emissions data
        geometry = emissions_gdf[['I_CELL','J_CELL','geometry']]\
            .copy().drop_duplicates().reset_index(drop=True)
        emissions_data = pd.DataFrame(emissions_gdf.drop(columns='geometry'))
        
        # Separately save the coordinate reference system
        crs = emissions_gdf.crs
        
        return geometry, emissions_data, crs
    
    def load_csv(self):
        ''' 
        Loads emissions data from a csv file.
        
        Requirements:
            - Emissions data must have the columns LAT_WGS84 and LON_WGS84 and
              five columns of pollutant data
        
        '''
        # Add log statements
        verboseprint(self.verbose, '- [EMISSIONS] Reading emissions from a CSV file.',
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # CSVs are read using pandas
        emissions_df = pd.read_csv(self.file_path, header=9)
        
        # Create geopandas geodataframe using lat/lon coordinates
        emissions_gdf = gpd.GeoDataFrame(emissions_df, 
                                         geometry = gpd.points_from_xy(emissions_df.LON_WGS84,
                                                                       emissions_df.LAT_WGS84), 
                                         crs='EPSG:4326')
        
        # Add unique identifiers for I_CELL and J_CELL
        emissions_gdf['I_CELL'] = np.arange(emissions_gdf.shape[0])
        emissions_gdf['J_CELL'] = np.arange(emissions_gdf.shape[0])
        
        # Remove nans
        emissions_gdf = emissions_gdf.fillna(0.0)
        
        # Split off geometry from emissions data
        geometry = emissions_gdf[['I_CELL','J_CELL','geometry']]\
            .copy().drop_duplicates().reset_index(drop=True)
        emissions_data = pd.DataFrame(emissions_gdf.drop(columns='geometry'))
        
        # Separately save the coordinate reference system
        crs = emissions_gdf.crs
        
        # Save a copy of the emissions shapefile 
        emissions_gdf_out_path = path.join(self.output_dir, 'shapes', self.f_out+'_emissions_input.shp')
        emissions_gdf.to_file(emissions_gdf_out_path)
        verboseprint(self.verbose, '- [EMISSIONS] Saved a copy of the emissions input as a shapefile: {}'.format(emissions_gdf_out_path),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        return geometry, emissions_data, crs
    

    def check_height(self):
        ''' Checks to see if a height column exists, otherwise adds groundlevel height '''
        if 'HEIGHT_M' in self.emissions_data.columns:
            return
        
        else:
            self.emissions_data['HEIGHT_M'] = 0.0
            verboseprint(self.verbose, '* [EMISSIONS] No height column was detected in the emissions data, so one was manually added.',
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))

        return

    def check_emissions(self):
        ''' Runs a number of checks to make sure that emissions data is valid  '''
        ## (TEST 1) Check for I_CELL and J_CELL
        has_indices = ('I_CELL' in self.emissions_data.columns) & ('J_CELL' in self.emissions_data.columns)
        
        ## (TEST 2) Check that columns are correct, or replace them if incorrect
        # Define the set of correct pollutants
        correct_pollutants = ['PM25', 'NH3', 'VOC', 'NOX', 'SOX']
        
        # Define a few dummy variables for storing information
        missing = []
        
        # Iterate through the list of correct pollutants to find and address issues
        for pol in correct_pollutants:
            if pol in self.emissions_data.columns: # True if correct pollutant is in the emissions_data
                pass # No action necessary
            else: # Enter the missing pollutant decision tree if missing
                choice=self.map_pollutant_name(pol) # Returns the missing pollutant if there is no suitable replacement
                if choice != pol:
                    self.emissions_data.rename(columns={choice:pol}, inplace=True) # Updates the emissions data
                else:
                    missing.append(pol) # Do not spit out error and exit yet
                    
        # After iterating through all of the pollutants, check which couldn't be identified or mapped
        if len(missing)>0:
            logging.info('\n<< [EMISSIONS] ERROR: required pollutants are missing from data. Please check data and try again. >>')
            logging.info('* [EMISSIONS] The tool requires that these five pollutants be included as columns in emissions data:', ', '.join(correct_pollutants))
            logging.info('* [EMISSIONS] The tool was not able to find or map:', ', '.join(missing))
            sys.exit()
            
        else: # All five pollutants are accounted for
            has_five_pollutants = True

        return has_indices and has_five_pollutants
    
    def map_pollutant_name(self, missing_pol):
        ''' If a pollutant name is not found in the emissions data, tries to map it before quitting '''
        # Below, there are a number of potential erroneous column names for each pollutant
        # These are ordered for most appropriate
        possible_wrong_names_dict = {'PM25':['PM2.5', 'PM2_5'],
                                     'NH3':['NH_3'],
                                     'VOC':['ROG', 'TOG', 'OG'],
                                     'NOX':['NO_X', 'NO2', 'NO_2'],
                                     'SOX':['SO_X', 'SO2', 'SO_2']}
        
        # Cut the dictionary into just the relevant ones for missing_pol
        possible_wrong_names = possible_wrong_names_dict[missing_pol]
        first_choice = 'na'
        
        # Iterate through the list of potential wrong names, stop if one is found
        for choice in possible_wrong_names:
            if choice in self.emissions_data.columns:
                logging.info('* [EMISSIONS] {} was not found in the emissions data. The program will continue using {} as a replacement.'.format(missing_pol, choice))
                first_choice = choice
                break
        
        # If nothing is found, return just the name of the missing pollutant
        if first_choice == 'na':
            return missing_pol
        
        # Otherwise, return the first choice
        else:
            return first_choice
        
    def filter_emissions(self, emissions):
        ''' Filters emissions based on inputted dictionary filter_dict '''
        filter_dict = self.filter_dict
        
        for key in filter_dict.keys():
            emissions = emissions.loc[emissions[key].isin(filter_dict[key]),:]
        
        return emissions
    
    def check_geo_types(self):
        ''' Checks for different geometry types and updates to polygons if needed '''
        # Check for non-polygon types and filter emissions into two sections
        polygon_filter = self.geometry.geom_type.str.upper().str.contains('POLYGON') # captures polygons and multi-polygons
        polygons = self.geometry[polygon_filter]
        
        # Speed this up by skipping if no calculations need to be done
        if polygons.shape[0] == self.geometry.shape[0]:
            return
        
        else: # Will need to do buffering on all other objects
            # Get just the non-polygon rows    
            if self.verbose:
                logging.info('* [EMISSIONS] {} non-polygon emissions sources identified. Adding a 0.005 m buffer to create polygons.'.format(self.geometry.shape[0]-polygons.shape[0]))
            non_polygons = self.geometry[~polygon_filter]
            new_polygons = self.buffer_emis(non_polygons, 0.005)
        
            # Update self.geometry to include these new ones
            self.geometry[~polygon_filter] = new_polygons
            
            return 
    
    def check_polygon_area(self):
        # Calculate polygon areas in square meters
        self.geometry['AREA_M2'] = self.geometry['geometry'].area
        
        # Convert square meters to square kilometers
        self.geometry['AREA_KM2'] = self.geometry['AREA_M2'] / 10**6
        
        # Get the maximum polygon area
        max_area = self.geometry['AREA_KM2'].max()
        
        # Check if maximum area exceeds 2500 km^2
        if max_area > 2500:
            logging.info('* [EMISSIONS] Large area emissions detected ({:.2f} km2). Consider carefully if results should be used for disparity, equity, and environmental justice analyses.'.format(max_area))

    def buffer_emis(self, emis_non_poly, dist):
        ''' Adds a buffer (in m) to the non-polygon type geometries in order to create polygons '''
        emis_non_poly['geometry'] = emis_non_poly.buffer(dist)
        return emis_non_poly

    
    def clean_up(self, details_to_keep, func=np.sum):
        ''' Simplifies emissions file by reducing unnecessary details '''
        # Start by deep-copying to avoid any accidental overwriting
        emissions_data_tmp = self.emissions_data.copy(deep=True)
        
        # If filter is enabled, filter emissions
        if self.filter:
            emissions_data_tmp = self.filter_emissions(emissions_data_tmp)
        
        # Use pandas processing to perform a groupby
        groupby_features = ['I_CELL', 'J_CELL', 'HEIGHT_M']+details_to_keep
        emissions_data_clean = emissions_data_tmp.groupby(groupby_features).agg(func)
        verboseprint(self.verbose, '- [EMISSIONS] Emissions have been reduced to contain the {} of emissions for each {}'.format(func.__name__, ', '.join(groupby_features)),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Clean up indices
        emissions_data_clean = emissions_data_clean.reset_index()
        
        # Scale emissions to ug/s
        scaling_factor = self.convert_units()
        emissions_data_clean[['PM25', 'NH3', 'VOC', 'NOX', 'SOX']] *= scaling_factor
        if scaling_factor != 1.0 and self.verbose:
            verboseprint(self.verbose, '- [EMISSIONS] Scaled emissions data by a factor of {:e} to convert from {} to ug/s.'.format(scaling_factor, self.units),
                         self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Limit columns
        emissions_data_clean = emissions_data_clean[groupby_features+['PM25', 'NH3', 'VOC', 'NOX', 'SOX']]
        
        return emissions_data_clean
    
    def convert_units(self):
        ''' Based on provided units, converts all emissions columns to ug/s '''
        # Import the units and the unit conversions
        units = self.units.lower() # force lower for simplicity
        unit_mass, unit_time = units.split('/')
        mass_units, time_units = self.get_unit_conversions()
        
        # Determine the scaling factor based on the provided units
        scaling_factor = mass_units[unit_mass]/time_units[unit_time]
        
        return scaling_factor
        
    
    def split_pollutants(self, emissions_object, pollutant, details_to_keep):
        ''' Creates separate objects for a pollutant '''
        # Start by deep-copying to avoid any accidental overwriting
        emissions_data_tmp = emissions_object.copy(deep=True)
        
        # Pull only the I_CELL, J_CELL, pollutant emissions, and other details
        groupby_features = ['I_CELL','J_CELL','HEIGHT_M',pollutant]+details_to_keep
        pollutant_emissions = emissions_data_tmp[groupby_features].copy().drop_duplicates().reset_index()
        
        # Add geometry back in
        pollutant_emissions = self.geometry.merge(pollutant_emissions, left_on=['I_CELL','J_CELL'], right_on=['I_CELL','J_CELL'])
        
        # Rename emissions column to 'Emissions_ug/s' for easier functionality later
        pollutant_emissions.rename(columns={pollutant:'EMISSIONS_UG/S'}, inplace=True)
        pollutant_emissions.drop(columns=['index'], inplace=True)

        # Prettier pollutant names
        pollutant_names = {'PM25':'primary PM2.5',
                           'NH3':'NH3',
                           'NOX':'NOx',
                           'VOC':'VOCs',
                           'SOX':'SOx'}
        
        verboseprint(self.verbose, '- [EMISSIONS] Successfully created emissions object for {} for {}.'.format(self.emissions_name, pollutant_names[pollutant]),
                     self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        return pollutant_emissions
    
    def which_layers(self):
        ''' Function that returns True or False for each of the layers '''
        # Get the unique values of the height column (faster than using all cells)
        heights = self.emissions_data_clean['HEIGHT_M'].unique()
        
        # Test the bounds of each layer
        L0_flag = sum(heights<57.0) > 0
        L1_flag = sum((heights>=57.0)&(heights<=140.0)) > 0
        L2_flag = sum(heights>=760.0) > 0
        isrm_hole_flag = sum((heights>140.0)&(heights<760.0)) > 0 
        
        # Return a warning message for the isrm_hole
        if isrm_hole_flag:
            logging.info('* [EMISSIONS] Emissions were identified with release heights between 140 - 760 meters.')
            logging.info('* The current version of the ISRM has not been trained on release heights in this range.')
            logging.info('* As a short-term solution, concentrations from these emissions will be estimated using\n  the average of the middle and high release layers.')
            logging.info('* For more information, please visit https://echo-air-model.github.io/docs/other_information/isrm_hole_patch.html.')
        
        return L0_flag, L1_flag, L2_flag, isrm_hole_flag
    
    def visualize_emissions(self, emissions_object, pollutant_name=''):
        ''' Creates map of emissions using simple chloropleth '''
        fig, ax = plt.subplots(1,1)
        emissions_object.plot(column='EMISSIONS_UG/S',
                              figsize=(20,10),
                              legend=True,
                              legend_kwds={'label':r'Emissions ($\mu$g/s)'},
                              ax = ax)
        ax.set_title(pollutant_name+' Emissions')
        fig.tight_layout()
        return fig
    
    def get_pollutant_layer(self, pol_name):
        ''' Returns pollutant layer '''        
        # Define a pollutant dictionary for convenience
        pollutant_dict = {'PM25':self.PM25,
                         'NH3':self.NH3,
                         'VOC':self.VOC,
                         'NOX':self.NOX,
                         'SOX':self.SOX}
        
        # Confirm pol_name is valid
        assert pol_name in pollutant_dict.keys()
        
        # Return pollutant layer
        return pollutant_dict[pol_name]
