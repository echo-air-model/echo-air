#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Census Object

@author: amyryao
last modified: 2024-08-16
"""

# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
import re
import os
from os import path
import logging
from tool_utils import *

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

class census:
    '''
    Defines a new object for storing and manipulating census data.
    
    INPUTS:
        - codebook_fp: the file path of the codebook data
        - tractdata_fp: the file path of the tract data
        - ipums_shp_fp: the file path of the shapefile data
        - sort_field: the field by which the data would be sorted by 
        - output_dir: a string pointing to the output directory
        - f_out: a string containing the filename pattern to be used in output files
        - verbose: a Boolean indicating whether or not detailed logging statements should be printed
        - debug_mode: a Boolean indicating whether or not to output debug statements
       
          
    CALCULATES:
        - combined_codebook: a dictionary mapping the combined codebook data
        - ca_tract_data: filtered California tract data
        - ca_tract_melt: melted California tract data for easier manipulation
        - age_mapper: a dictionary mapping age descriptions to age bins
        - ca_tract_sum: summarized population data by GISJOIN, YEAR, RACE/ETHNICITY, and AGE_BIN
        - ca_tracts: merged geographic and census data
        - geo_data: geographic data
        - ca_tract_pivot: pivoted data for easier analysis
        - ca_tracts_to_export: final processed data ready for export as a feather file      
    '''

    def __init__(self, codebook_fp, tractdata_fp, ipums_shp_fp, output_dir, f_out, sort_field='RACE/ETHNICITY', verbose=False, debug_mode=False):
        self.codebook_fp = codebook_fp
        self.tractdata_fp = tractdata_fp
        self.ipums_shp_fp = ipums_shp_fp
        self.output_dir = output_dir
        self.sort_field = sort_field
        self.f_out = f_out
        self.verbose = verbose
        self.debug_mode = debug_mode

        # Read through files
        self.codebook = self.load_codebook()
        self.tract_data = pd.read_csv(tractdata_fp, low_memory=False)
        self.census_geo = gpd.read_file(ipums_shp_fp)

    def load_codebook(self):
        '''
        Loads the codebook file and parses it into a dictionary.
        
        RETURNS:
            - codebook: a dictionary mapping column headers to their descriptions
        '''

        regex_str = r'\s{8}[A-Z0-9]+:\s*.+'
        
        # Set up objects for doing this
        with open(self.codebook_fp) as codebook_file:
            codebook_txt = codebook_file.readlines()
        codebook = {}

        # Loop through the codebook text file
        for line in codebook_txt:
            if re.match(regex_str, line) is not None:
                
                # Read in the line and split it up based on colon
                split_line = line.strip(' ').strip('\n').split(':')
                
                # Split this into header and label
                header = split_line[0]
                label = split_line[1:]
               
                # Some of the labels have a colon within, so deal with this
                new_label = label[0] + ':' + label[1] if len(label) > 1 else label[0]
                
                # Clean up the label a bit
                label = new_label.strip(' ')
                # Combine and add to dictionary
                codebook[header] = label

        # For simplicity, do this through a for-loop
        for code in codebook.keys():
            
            # Get the value from the codebook
            tmp = codebook[code]
            
            # Split the value and see the length, if 2 then it's hispanic/latino
            if len(tmp.split('>>')) == 2:
                codebook[code] = 'Hispanic/Latino >> '+tmp
        verboseprint(self.verbose, '- [CENSUS] Codebook loaded successfully', self.debug_mode, frameinfo=getframeinfo(currentframe()))
        return codebook

    def process_codebook(self):
        '''
        Processes the loaded codebook to create a combined codebook with race codes.
        
        RETURNS:
            - combined_codebook: a dictionary mapping combined code descriptions
        '''
        verboseprint(self.verbose, '- [CENSUS] Processing the codebook', self.debug_mode, frameinfo=getframeinfo(currentframe()))
    
        # Set up regex codes for the desired fields
        universe_str = r'\s{4}[A-Z][a-z]{7}:'
        nhgis_code_str = r'\s{4}[A-Z]{5}\s[a-z]{4}'

        # Create a dictionary for this
        race_codes = {}

        # Set up flags to get the NHGIS code that follows the universe description
        universe_flag, nhgis_flag = 0, 0

        # Set up objects for doing this
        with open(self.codebook_fp) as codebook_file:
            codebook_txt = codebook_file.readlines()

        # Loop through the codebook text file
        for line in codebook_txt:
            # Check if the universe field is matched
            if re.match(universe_str, line) is not None:
                universe = line.split(':')[1].strip(' ').strip('\n')[15:]
                universe_flag = 1 
            
            # Check if NHGIS Code field is matched
            if re.match(nhgis_code_str, line) is not None:
                nhgis = line.split(':')[1].strip(' ').strip('\n')
                nhgis_flag = 1 

            # If both are found, store in dictionary and reset
            if universe_flag == 1 and nhgis_flag == 1:
                race_codes[nhgis] = universe
                universe_flag, nhgis_flag = 0, 0 #resets

        # Iterate through and chop off everything after the comma
        race_codes_updated = {code: race_codes[code].split(',')[0] for code in race_codes.keys()}

        
        # Create a combined codebook dictionary
        combined_codebook = {}


        # Loop through the items in the codebook dictionary
        for item in self.codebook.items():
            # Grab basic information from the codebook dictionary
            header, no_race_value = item[0], item[1]

            # Look for the key in the race_codes_updated dictionary            
            if header[0:3] in race_codes_updated.keys():
                nhgis_code_desc = race_codes_updated[header[0:3]]
                combined_codebook[header] = nhgis_code_desc + ': ' + self.codebook[header]
            else: #Fields like GISJOIN are not in race_codes_updated
                combined_codebook[header] = no_race_value

        verboseprint(self.verbose, '- [CENSUS] Codebook processed successfully', self.debug_mode, frameinfo=getframeinfo(currentframe()))
        return combined_codebook

    # Depending on the year, the description is split using >> or : 
    def split_description(self, description):
        '''Function to split based on ':', and if ':' is not found, split based on '>>' '''
        if isinstance(description, str):
            if '>>' in description:
                return description.split('>>')
            elif ':' in description:
                return description.split(':')
            else:
                return [description]
        else:
            return [str(description)]

    
    def preprocess_data(self):
        '''
        Preprocesses the census data by filtering, melting, mapping age bins, and merging with geographic data.
        Saves the processed data to the specified output file.
        '''
        # Return print statement to indicate the preprocessing has begun
        verboseprint(self.verbose, '- [CENSUS] Processing data', self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Processes the loaded codebook to create a combined codebook with race codes
        combined_codebook = self.process_codebook()
        
        # Define a list of columns to drop
        cols_to_drop = ['REGIONA', 'DIVISIONA', 'STATEA', 'COUNTYA', 'COUSUBA', 'PLACEA', 'TRACTA', 'CONCITA', 'AIANHHA', 
                        'RES_ONLYA', 'TRUSTA', 'AITSCEA', 'TTRACTA', 'ANRCA', 'CBSAA', 'METDIVA', 'CSAA', 'NECTAA', 
                        'NECTADIVA', 'CNECTAA', 'UAA', 'URBRURALA', 'CDA', 'SLDUA', 'SLDLA', 'ZCTA5A', 'SUBMCDA', 
                        'SDELMA', 'SDSECA', 'SDUNIA', 'NAME']

        
        # Filter the dataframe
        tract_data_small = self.tract_data.loc[:, ~self.tract_data.columns.isin(cols_to_drop)]
        
        #Trim the tract_data_small dataset
        ca_tract_data = tract_data_small[tract_data_small['STATE'] == 'California'].copy()
       
        verboseprint(self.verbose, '- [CENSUS] Data filtered for California.', self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        # Get a list of the melt values
        melt_values = list(ca_tract_data.columns)[4:]

        # Perform melt
        ca_tract_melt = ca_tract_data.melt(value_vars=melt_values, id_vars=['GISJOIN','YEAR'], var_name='GROUP CODE', value_name='POPULATION').reset_index(drop=True)

        # Map the code in the new GROUP CODE column to the descriptive names in the codebook object.
        ca_tract_melt['GROUP DESC'] = ca_tract_melt['GROUP CODE'].map(combined_codebook)
        
        # Add a dummy column to filter out the rows with totals
        ca_tract_melt['DUMMY'] = ca_tract_melt['GROUP DESC'].apply(lambda x: len(self.split_description(x)))
        
        # Filter out any rows where DUMMY is not 3
        ca_tract_melt = ca_tract_melt[ca_tract_melt['DUMMY'] == 3].copy()
        ca_tract_melt = ca_tract_melt[['GISJOIN', 'YEAR', 'POPULATION', 'GROUP DESC']].reset_index(drop=True)
        
        # Add a column based on the sort_field using the str.split method
        ca_tract_melt[self.sort_field] = ca_tract_melt['GROUP DESC'].apply(lambda x: self.split_description(x)[0])
        
        # Add an Age column based using the str.split method
        ca_tract_melt['AGE'] = ca_tract_melt['GROUP DESC'].apply(lambda x: self.split_description(x)[2])
        
        verboseprint(self.verbose, '- [CENSUS] Data melted and columns created for {}. and AGE' .format(self.sort_field), self.debug_mode, frameinfo=getframeinfo(currentframe()))
       
        # Unique values of age
        unique_ages = ca_tract_melt['AGE'].unique()

        # This will be our mapping dictionary
        age_mapper = {}

        for age in unique_ages:
            
            # Gets ride of white spaces
            age = age.strip(' ')

            # Check if the age has only two entries (in which case it is probably # years)
            if len(age.split(' ')) == 2:
                # Get the age number as n
                n = float(age.split(' ')[0])
                
                # Pull lower and upper bins using math
                lower_bin = int(np.floor(n / 5.0) * 5.0)
                upper_bin = int(lower_bin + 4)
                
                # Make an update if the upper bin is 0, it should be 1
                if lower_bin == 0:
                    lower_bin = 1
                
                # Make an update if the lower_bin is 85 or over
                if upper_bin >= 85:
                    age_mapper[age] = '85UP'
                else:
                    age_mapper[age] = str(lower_bin) + 'TO' + str(upper_bin)
            
            # Add in the babies
            elif age == 'Under 1 year':
                age_mapper[age] = '0TO0'
            else: # Everyone else is over 85
                age_mapper[age] = '85UP'
        
        # Map this onto the dataframe
        ca_tract_melt['AGE_BIN'] = ca_tract_melt['AGE'].str.strip(' ').map(age_mapper)
  
        # Perform a groupby sum
        ca_tract_sum = ca_tract_melt.groupby(['GISJOIN', 'YEAR', self.sort_field, 'AGE_BIN'])['POPULATION'].sum().reset_index()
        
        # Merge with the geodata
        ca_tracts = self.census_geo.merge(ca_tract_sum, on='GISJOIN', how='right')

        # Simplify and clean up columns
        ca_tracts = ca_tracts[['GISJOIN', self.sort_field,'AGE_BIN','YEAR','POPULATION','geometry']].copy()

        verboseprint(self.verbose, '- [CENSUS] Summarized data and merged with geographic data.', self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        if self.sort_field == 'RACE/ETHNICITY': 
            # Remove specific prefixes from the 'RACE/ETHNICITY' column
            ca_tracts['RACE/ETHNICITY'] = ca_tracts['RACE/ETHNICITY'].str.replace(' Latino Persons: ', '').str.replace('ino Persons: ', '')
        
        # Strip any leading or trailing whitespace
        ca_tracts[self.sort_field] = ca_tracts[self.sort_field].str.strip()
        
        if self.sort_field == 'RACE/ETHNICITY': 
            # Define race/ethnicity mapper with exceptions
            race_eth_mapper = {
                'American Indian and Alaska Native alone': 'INDIG',
                'Asian alone': 'ASIAN',
                'Black or African American alone': 'BLACK',
                'Hispanic or Latino': 'HISLA',
                'Hispanic/Latino': 'HISLA',  # Note the difference here
                'Native Hawaiian and Other Pacific Islander alone': 'PACIS',
                'Some Other Race alone': 'OTHER',
                'Some other race alone': 'OTHER',  # Note the difference here
                'Two or more races': 'OTHER',  # Note the difference here
                'Two or More Races': 'OTHER',
                'White alone': 'WHITE'
            }

            # Map using race_eth_mapper
            ca_tracts['GROUP'] = ca_tracts[self.sort_field].map(race_eth_mapper)
        else: 
            ca_tracts['GROUP'] = ca_tracts[self.sort_field]

        # Drop rows where 'GROUP' column has NaN values
        ca_tracts = ca_tracts.dropna(subset=['GROUP'])

        # Set display options to show all columns
        pd.set_option('display.max_columns', None)

        # Grab geodata and store as temporary dataframe
        geo_data = ca_tracts[['GISJOIN','geometry']].drop_duplicates().copy()
        
        # Perform a groupby sum to get one value per combination of demographic information
        ca_tracts_tmp = ca_tracts[['GISJOIN','GROUP','AGE_BIN','YEAR','POPULATION']].copy()
        ca_tracts_tmp = ca_tracts_tmp.groupby(['GISJOIN','GROUP','AGE_BIN','YEAR'])['POPULATION'].sum().reset_index()

      
        
        # Recombine with geodata
        ca_tracts = geo_data.merge(ca_tracts_tmp, on='GISJOIN', how='right')

        # Add the START_AGE and END_AGE columns
        ca_tracts['START_AGE'] = ca_tracts.apply(lambda x: self.get_start_age(x['AGE_BIN']), axis=1)
        ca_tracts['END_AGE'] = ca_tracts.apply(lambda x: self.get_end_age(x['AGE_BIN']), axis=1)
        
        # Split out the geometry and add POP_ID
        ca_tract_geo = ca_tracts[['GISJOIN', 'geometry']].drop_duplicates().copy().reset_index(drop=True)
        ca_tract_geo['POP_ID'] = ca_tract_geo.index

        # Perform a pivot on the ca_tracts data without the geographic information
        ca_tract_tmp = ca_tracts[['GISJOIN', 'YEAR', 'GROUP', 'AGE_BIN', 'START_AGE', 'END_AGE', 'POPULATION']].copy()
        ca_tract_pivot = ca_tract_tmp.pivot(index=['YEAR','GISJOIN','AGE_BIN','START_AGE','END_AGE'], columns='GROUP',
                                         values='POPULATION').reset_index()
            
      
        
        # Add a TOTAL column
        ca_tract_pivot['TOTAL'] = ca_tract_pivot[['ASIAN', 'BLACK', 'HISLA', 'INDIG', 'OTHER', 'PACIS', 'WHITE']].sum(axis=1)

        # Confirm the total is still correct
        ca_tract_pivot['TOTAL'].sum()

        # Merge with the geodata that was stored
        ca_pop_data = pd.merge(ca_tract_geo, ca_tract_pivot, on='GISJOIN')

        

        # Do final cleanup
        ca_tracts_to_export = ca_pop_data[['POP_ID', 'YEAR', 'AGE_BIN', 'START_AGE', 'END_AGE', 'TOTAL', 'ASIAN', 'BLACK',
                                   'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER', 'geometry']].copy()  

        # Creating the file name
        fname = self.f_out + '.feather'
        fpath = os.path.join(self.output_dir, fname)

        # Logging statement for exporting
        logging.info('- [CENSUS] Stored at: {}'.format(fpath))
        
        # Save to file
        ca_tracts_to_export.to_feather(fpath)

        verboseprint(self.verbose, '- [CENSUS] Data preprocessing complete and saved to {}.'.format(fpath), self.debug_mode, frameinfo=getframeinfo(currentframe()))
        
        return
        
    # Define helper functions for adding the start and end ages
    def get_start_age(self, age_bin):
        ''' Gets the start age from the age_bin '''
        try:
            start_age = int(age_bin.split('TO')[0])
        except:
            start_age = 85
        return start_age
    
    def get_end_age(self,age_bin):
        ''' Gets the end age from the age_bin '''
        try:
            end_age = int(age_bin.split('TO')[1])
        except:
            end_age = 99
        return end_age
