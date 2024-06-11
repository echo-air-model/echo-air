#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Data Object

@author: libbykoolik
last modified: 2024-06-11
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

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

class census:
    '''
    Defines a new object for storing and manipulating census data.
    
    INPUTS:
        - codebook_fp: the file path of the codebook data
        - tractdata_fp: the file path of the tract data
        - ipums_shp_fp: the file path of the shapefile data
        - out_file: the file path where the processed data will be saved
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
          
    EXTERNAL FUNCTIONS:
        - get_start_age: helper function to extract the start age from age_bin
        - get_end_age: helper function to extract the end age from age_bin
    '''

    def __init__(self, codebook_fp, tractdata_fp, ipums_shp_fp, out_file, verbose=False, debug_mode=False):
        self.codebook_fp = codebook_fp
        self.tractdata_fp = tractdata_fp
        self.ipums_shp_fp = ipums_shp_fp
        self.out_file = out_file
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.verboseprint = logging.info if self.verbose else lambda *a, **k: None
        self.codebook = self.load_codebook()
        self.tract_data = pd.read_csv(tractdata_fp, encoding="ISO-8859-1")
        self.census_geo = gpd.read_file(ipums_shp_fp)

    def load_codebook(self):
        '''
        Loads the codebook file and parses it into a dictionary.
        
        RETURNS:
            - codebook: a dictionary mapping column headers to their descriptions
        '''
        self.verboseprint("Loading codebook from file: {}".format(self.codebook_fp), self.debug_mode)
        regex_str = r'\s{8}[A-Z0-9]+:\s*.+'
        codebook_file = open(self.codebook_fp)
        codebook_txt = codebook_file.readlines()
        codebook = {}

        for line in codebook_txt:
            if re.match(regex_str, line) is not None:
                split_line = line.strip(' ').strip('\n').split(':')
                header = split_line[0]
                label = split_line[1:]
                if len(label) > 1:
                    new_label = label[0] + ':' + label[1]
                else:
                    new_label = label[0]
                label = new_label.strip(' ')
                codebook[header] = label

        self.verboseprint("Codebook loaded successfully.", self.debug_mode)
        return codebook

    def process_codebook(self):
        '''
        Processes the loaded codebook to create a combined codebook with race codes.
        
        RETURNS:
            - combined_codebook: a dictionary mapping combined code descriptions
        '''
        self.verboseprint("Processing the codebook...", self.debug_mode)
        universe_str = r'\s{4}[A-Z][a-z]{7}:'
        nhgis_code_str = r'\s{4}[A-Z]{5}\s[a-z]{4}'
        race_codes = {}
        universe_flag, nhgis_flag = 0, 0

        codebook_file = open(self.codebook_fp)
        codebook_txt = codebook_file.readlines()

        for line in codebook_txt:
            if re.match(universe_str, line) is not None:
                universe = line.split(':')[1].strip(' ').strip('\n')[15:]
                universe_flag = 1 

            if re.match(nhgis_code_str, line) is not None:
                nhgis = line.split(':')[1].strip(' ').strip('\n')
                nhgis_flag = 1 

            if universe_flag == 1 and nhgis_flag == 1:
                race_codes[nhgis] = universe
                universe_flag, nhgis_flag = 0, 0 

        race_codes_updated = {code: race_codes[code].split(',')[0] for code in race_codes.keys()}

        combined_codebook = {}
        for item in self.codebook.items():
            header, no_race_value = item[0], item[1]
            if header[0:3] in race_codes_updated.keys():
                nhgis_code_desc = race_codes_updated[header[0:3]]
                combined_codebook[header] = nhgis_code_desc + ': ' + self.codebook[header]
            else:
                combined_codebook[header] = no_race_value

        self.verboseprint("Codebook processed successfully.", self.debug_mode)
        return combined_codebook

    def preprocess_data(self):
        '''
        Preprocesses the census data by filtering, melting, mapping age bins, and merging with geographic data.
        Saves the processed data to the specified output file.
        '''
        self.verboseprint("Starting data preprocessing...", self.debug_mode)
        combined_codebook = self.process_codebook()
        
        cols_to_drop = ['REGIONA', 'DIVISIONA', 'STATEA', 'COUNTYA', 'COUSUBA', 'PLACEA', 'TRACTA', 'CONCITA', 'AIANHHA', 
                        'RES_ONLYA', 'TRUSTA', 'AITSCEA', 'TTRACTA', 'ANRCA', 'CBSAA', 'METDIVA', 'CSAA', 'NECTAA', 
                        'NECTADIVA', 'CNECTAA', 'UAA', 'URBRURALA', 'CDA', 'SLDUA', 'SLDLA', 'ZCTA5A', 'SUBMCDA', 
                        'SDELMA', 'SDSECA', 'SDUNIA', 'NAME']

        tract_data_small = self.tract_data.loc[:, ~self.tract_data.columns.isin(cols_to_drop)]
        ca_tract_data = tract_data_small[tract_data_small['STATE'] == 'California'].copy()
        
        self.verboseprint("Data filtered for California.", self.debug_mode)

        melt_values = list(ca_tract_data.columns)[4:]
        ca_tract_melt = ca_tract_data.melt(value_vars=melt_values, id_vars=['GISJOIN', 'YEAR'], var_name='GROUP CODE', value_name='POPULATION').reset_index(drop=True)
        ca_tract_melt['GROUP DESC'] = ca_tract_melt['GROUP CODE'].map(combined_codebook)
        ca_tract_melt['DUMMY'] = ca_tract_melt['GROUP DESC'].str.split(':').str.len()
        ca_tract_melt = ca_tract_melt[ca_tract_melt['DUMMY'] == 3].copy()
        ca_tract_melt = ca_tract_melt[['GISJOIN', 'YEAR', 'POPULATION', 'GROUP DESC']].reset_index(drop=True)
        ca_tract_melt['RACE/ETHNICITY'] = ca_tract_melt['GROUP DESC'].str.split(':').str[0]
        ca_tract_melt['AGE'] = ca_tract_melt['GROUP DESC'].str.split(':').str[2]

        self.verboseprint("Data melted and columns created for RACE/ETHNICITY and AGE.", self.debug_mode)

        unique_ages = ca_tract_melt['AGE'].unique()
        age_mapper = {}

        for age in unique_ages:
            age = age.strip(' ')
            if len(age.split(' ')) == 2:
                n = float(age.split(' ')[0])
                lower_bin = int(np.floor(n / 5.0) * 5.0)
                upper_bin = int(lower_bin + 4)
                if lower_bin == 0:
                    lower_bin = 1
                if upper_bin >= 85:
                    age_mapper[age] = '85UP'
                else:
                    age_mapper[age] = str(lower_bin) + 'TO' + str(upper_bin)
            elif age == 'Under 1 year':
                age_mapper[age] = '0TO0'
            else:
                age_mapper[age] = '85UP'

        ca_tract_melt['AGE_BIN'] = ca_tract_melt['AGE'].str.strip(' ').map(age_mapper)
        ca_tract_sum = ca_tract_melt.groupby(['GISJOIN', 'YEAR', 'RACE/ETHNICITY', 'AGE_BIN'])['POPULATION'].sum().reset_index()

        ca_tracts = self.census_geo.merge(ca_tract_sum, on='GISJOIN', how='right')
        ca_tracts = ca_tracts[['GISJOIN', 'RACE/ETHNICITY', 'AGE_BIN', 'YEAR', 'POPULATION', 'geometry']].copy()
        
        self.verboseprint("Summarized data and merged with geographic data.", self.debug_mode)

        race_eth_mapper = {'American Indian and Alaska Native alone ': 'INDIG',
                           'Asian alone ': 'ASIAN',
                           'Black or African American alone ': 'BLACK',
                           'Hispanic/Latino ': 'HISLA',
                           'Native Hawaiian and Other Pacific Islander alone ': 'PACIS',
                           'Some other race alone ': 'OTHER',
                           'Two or more races ': 'OTHER',
                           'White alone ': 'WHITE'}

        ca_tracts['GROUP'] = ca_tracts['RACE/ETHNICITY'].map(race_eth_mapper)

        geo_data = ca_tracts[['GISJOIN', 'geometry']].drop_duplicates().copy()
        ca_tracts_tmp = ca_tracts[['GISJOIN', 'GROUP', 'AGE_BIN', 'YEAR', 'POPULATION']].copy()
        ca_tracts_tmp = ca_tracts_tmp.groupby(['GISJOIN', 'GROUP', 'AGE_BIN', 'YEAR'])['POPULATION'].sum().reset_index()
        ca_tracts = geo_data.merge(ca_tracts_tmp, on='GISJOIN', how='right')

        ca_tracts['START_AGE'] = ca_tracts.apply(lambda x: self.get_start_age(x['AGE_BIN']), axis=1)
        ca_tracts['END_AGE'] = ca_tracts.apply(lambda x: self.get_end_age(x['AGE_BIN']), axis=1)

        ca_tract_geo = ca_tracts[['GISJOIN', 'geometry']].drop_duplicates().copy().reset_index(drop=True)
        ca_tract_geo['POP_ID'] = ca_tract_geo.index

        ca_tract_tmp = ca_tracts[['GISJOIN', 'YEAR', 'GROUP', 'AGE_BIN', 'START_AGE', 'END_AGE', 'POPULATION']].copy()
        ca_tract_pivot = ca_tract_tmp.pivot(index=['YEAR', 'GISJOIN', 'AGE_BIN', 'START_AGE', 'END_AGE'], columns='GROUP')
        ca_tract_pivot.columns = ca_tract_pivot.columns.droplevel(0)

        ca_tract_pivot['TOTAL'] = ca_tract_pivot[['ASIAN', 'BLACK', 'HISLA', 'INDIG', 'OTHER', 'PACIS', 'WHITE']].sum(axis=1)

        ca_pop_data = pd.merge(ca_tract_geo, ca_tract_pivot, on='GISJOIN')
        ca_tracts_to_export = ca_pop_data[['POP_ID', 'YEAR', 'AGE_BIN', 'START_AGE', 'END_AGE', 'TOTAL', 'ASIAN', 'BLACK', 'HISLA', 'INDIG', 'PACIS', 'WHITE', 'OTHER', 'geometry']].copy()
        ca_tracts_to_export.to_feather(self.out_file)

        self.verboseprint("Data preprocessing complete and saved to {}".format(self.out_file), self.debug_mode)

        return self.out_file

    @staticmethod
    def get_start_age(age_bin):
        ''' 
        Gets the start age from the age_bin.
        
        INPUTS:
            - age_bin: the age bin string
        
        RETURNS:
            - start_age: the starting age of the bin
        '''
        try:
            start_age = int(age_bin.split('TO')[0])
        except:
            start_age = 85
        return start_age

    @staticmethod
    def get_end_age(age_bin):
        ''' 
        Gets the end age from the age_bin.
        
        INPUTS:
            - age_bin: the age bin string
        
        RETURNS:
            - end_age: the ending age of the bin
        '''
        try:
            end_age = int(age_bin.split('TO')[1])
        except:
            end_age = 99
        return end_age
