#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Data Object

@author: libbykoolik
last modified: 2023-09-12
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
    
    
    def crosswalk(self,population_gdf,isrm_gdf,hia_flag):
        ''' Creates a crosswalk of the population cells and ISRM Grid cells. Returns a crosswalk geodataframe'''
        #Get unique geometries
        unique_pop = population_gdf.drop_duplicates(subset=["POP_ID", "geometry"])[["POP_ID", "geometry"]].copy()
        unique_pop = unique_pop[unique_pop.geometry.notnull()]
        unique_pop = gpd.GeoDataFrame(unique_pop, geometry="geometry", crs=population_gdf.crs)

        #Intersect once
        if unique_pop.crs != isrm_gdf.crs:
            unique_pop = unique_pop.to_crs(isrm_gdf.crs)
        intersection = gpd.overlay(unique_pop, isrm_gdf, how="intersection")

        #Area calcs
        intersection["area_intersection"] = intersection.geometry.area
        intersection["area_pop"] = intersection["POP_ID"].map(unique_pop.set_index("POP_ID").geometry.area.to_dict())
        intersection["area_isrm"] = intersection["ISRM_ID"].map(isrm_gdf.set_index("ISRM_ID").geometry.area.to_dict())
        intersection["fpop"] = intersection["area_intersection"] / intersection["area_pop"]
        intersection["fisrm"] = intersection["area_intersection"] / intersection["area_isrm"]

        # Re-merge with age bins (if hia_flag is True)
        if hia_flag:
            age_bins = population_gdf[["POP_ID", "AGE_BIN"]].drop_duplicates()
            crosswalk = intersection.merge(age_bins, on="POP_ID", how="left")
            crosswalk = crosswalk[["POP_ID", "AGE_BIN", "ISRM_ID", "fpop", "fisrm", "geometry"]].copy()
        else:
            crosswalk = intersection[["POP_ID", "ISRM_ID", "fpop", "fisrm", "geometry"]].copy()

        # Note that the only ISRM Grid IDs in this dataframe are ones that intersect with a district
        return crosswalk
    
    def allocate_pop(self,population_gdf,isrm_gdf,hia_flag):
        ''' Takes crosswalk and reallocates popualtion into the ISRM grid cells'''
        total_population = population_gdf["TOTAL"].sum()
        crosswalk_df = self.crosswalk(population_gdf, isrm_gdf, hia_flag)
        if hia_flag == True:
            #Merge all data
            merged_data = crosswalk_df.merge(population_gdf[ ['POP_ID', 'YEAR', 'AGE_BIN', 'START_AGE', 'END_AGE', 'TOTAL', 'ASIAN','BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']], on=["POP_ID","AGE_BIN"] , how="left")
            pop_columns = ['TOTAL', 'ASIAN','BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']

            #Multiply population counts by respective fractions
            for col in pop_columns:
                merged_data[f"{col}_adjusted"] = merged_data[col] * merged_data["fpop"]

            #aggregate by ISRM ID and AGE BIN
            isrm_group = merged_data.groupby(["ISRM_ID","AGE_BIN"])[['geometry','START_AGE', 'END_AGE','TOTAL_adjusted', 'ASIAN_adjusted','BLACK_adjusted', 'HISLA_adjusted', 'INDIG_adjusted', 'PACIS_adjusted', 'WHITE_adjusted', 'OTHER_adjusted']].agg({'geometry':'first', 'START_AGE':'first', 'END_AGE':'first','TOTAL_adjusted':'sum', 'ASIAN_adjusted':'sum','BLACK_adjusted':'sum', 'HISLA_adjusted':'sum', 'INDIG_adjusted':'sum', 'PACIS_adjusted':'sum', 'WHITE_adjusted':'sum', 'OTHER_adjusted':'sum'}).reset_index()

            #Create a list of all ISRM_ID, START_AGE, END_AGE options
            isrm_ids = isrm_gdf[['ISRM_ID']].drop_duplicates()
            age_bins = population_gdf[['AGE_BIN', 'START_AGE', 'END_AGE']].drop_duplicates()
            isrm_age_combos = isrm_ids.assign(key=1).merge(age_bins.assign(key=1), on='key').drop('key', axis=1)

            #Right join on this list so that all ISRM IDS and age_bins are present in the dataframe
            isrm_group = isrm_age_combos.merge(isrm_group, on=["ISRM_ID", "AGE_BIN"], how="left")

            #Merge again with orginal isrm dataframe to get original geometry
            isrm_group = isrm_group.merge(isrm_gdf, on = "ISRM_ID", how = 'right')
            isrm_group = isrm_group.drop(columns = {"AGE_BIN","geometry_x", "START_AGE_y", "END_AGE_y"})
            isrm_group = isrm_group.rename(columns={"START_AGE_x":"START_AGE","END_AGE_x":"END_AGE","geometry_y" : "geometry"})
            isrm_group = isrm_group.rename(columns={'TOTAL_adjusted':'TOTAL', 'ASIAN_adjusted':'ASIAN','BLACK_adjusted':'BLACK', 'HISLA_adjusted':'HISLA', 'INDIG_adjusted':'INDIG', 'PACIS_adjusted':'PACIS', 'WHITE_adjusted':'WHITE', 'OTHER_adjusted':"OTHER"})

            #Reorganize column order
            cols = isrm_group.columns.tolist()  # Get column names as a list
            last_col = cols.pop()  # Remove last column ('geometry')
            cols.insert(1, last_col)
            isrm_group = isrm_group[cols]
            isrm_group[cols] = isrm_group[cols].fillna(0)
            isrm_group = gpd.GeoDataFrame(isrm_group, geometry="geometry", crs=isrm_gdf.crs)
        else:
            #Merge all data
            merged_data = crosswalk_df.merge(population_gdf[['POP_ID', 'YEAR', 'TOTAL', 'ASIAN','BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']], on=["POP_ID"] , how="left")
            pop_columns = ['TOTAL', 'ASIAN','BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER']

            # Calculate fractions and multiply by population counts
            for col in pop_columns:
                merged_data[f"{col}_adjusted"] = merged_data[col] * merged_data["fpop"]

            #Aggregate by ISRM ID
            isrm_group = merged_data.groupby(["ISRM_ID"])[['geometry','TOTAL_adjusted', 'ASIAN_adjusted','BLACK_adjusted', 'HISLA_adjusted', 'INDIG_adjusted', 'PACIS_adjusted', 'WHITE_adjusted', 'OTHER_adjusted']].agg({'geometry':'first', 'TOTAL_adjusted':'sum', 'ASIAN_adjusted':'sum','BLACK_adjusted':'sum', 'HISLA_adjusted':'sum', 'INDIG_adjusted':'sum', 'PACIS_adjusted':'sum', 'WHITE_adjusted':'sum', 'OTHER_adjusted':'sum'}).reset_index()
            isrm_group = isrm_group.rename(columns={'TOTAL_adjusted':'TOTAL', 'ASIAN_adjusted':'ASIAN','BLACK_adjusted':'BLACK', 'HISLA_adjusted':'HISLA', 'INDIG_adjusted':'INDIG', 'PACIS_adjusted':'PACIS', 'WHITE_adjusted':'WHITE', 'OTHER_adjusted':"OTHER"})
            isrm_group = isrm_group.drop(columns = "geometry")

            #Remerge with ISRM Grid so that all ISRMs are present
            isrm_group = isrm_group.merge(isrm_gdf, on = "ISRM_ID", how = 'right')
            cols = isrm_group.columns.tolist()  # Get column names as a list
            last_col = cols.pop()  # Remove last column ('geometry')
            cols.insert(1, last_col)
            isrm_group = isrm_group[cols]
            isrm_group[cols] = isrm_group[cols].fillna(0)
            isrm_group = gpd.GeoDataFrame(isrm_group, geometry="geometry", crs=isrm_gdf.crs)

        final_population = isrm_group["TOTAL"].sum()
        assert np.isclose(total_population, final_population)

        return isrm_group

