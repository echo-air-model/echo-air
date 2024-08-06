#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control File Reading Object

@author: libbykoolik

last modified: 2024-01-19

"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from os import path
import sys
import re
import difflib

#%%
class pop_control_file:
    '''
    Defines a new object for reading the control file and storing information.
    
    INPUTS:
        - file_path: File path to the control file     
    
    ATTRIBUTES:
        - valid_structure: Boolean keyword based on internal checks of the control file format
    '''
    def __init__(self, file_path):
        ''' Initializes the control file object'''  
        logging.info('\n << Reading Population Control File >>')
        
        # Initialize path and check that it is valid
        self.file_path = file_path
        self.output_file_name, self.codebook_fp, self.tractdata_fp, self.ipums_shp_fp, self.verbose = self.get_all_inputs()
        self.valid_inputs = self.check_inputs()
        
        # If the inputs aren't all valid, the rest of the script doesn't run
        if self.valid_inputs: 
            logging.info('\n << Control file was successfully imported and inputs are correct >>')
        else: 
            logging.info('\n << ERROR: Control file was successfully imported but inputs are not correct >>')
            sys.exit('Exiting the program due to invalid inputs.')
    
    def get_input_value(self, keyword, upper=False):
        ''' Gets the input for the given keyword '''
        
        # Iterate through each line of the file to find the keyword
        for line in open(self.file_path):
            re_k = '- '+keyword+':' # Grabs exact formatting
            if re_k in line:
                line_val = line.split(':')[1].strip('\n').strip(' ')
            
        if upper: # Should be uppercased
            line_val = line_val.upper()
            
        return line_val
    
    def get_all_inputs(self):
        ''' Once it passes the basic control file checks, import the values '''
        output_file_name = self.get_input_value('OUTPUT_FILE_NAME')
        codebook_fp = self.get_input_value('CODEBOOK_FP')
        tractdata_fp = self.get_input_value('TRACTDATA_FP')
        ipums_shp_fp = self.get_input_value('IPUMS_SHP_FP')
        verbose = self.get_input_value('VERBOSE')

        return output_file_name, codebook_fp, tractdata_fp, ipums_shp_fp, verbose

    def check_inputs(self):
        ''' Checks validity of the control file before running''' 
        # This following is the current code with control file, but it returns true regardless since I believe the content is read in as a string
        # valid_batch_name = type(self.batch_name) == str
        # Would we like it so that if there's any numbers at all it shouldn't be valid or something of the sort? I can also change control file code then
        # This one currently doesn't allow all numbers: 
        valid_output_file_name = isinstance(self.output_file_name, str) and not self.output_file_name.isdigit()
        logging.info('* The output file name provided is not valid (must be string). ') if not valid_output_file_name else ''
        
        # If the output file name is blank, replace with "population"
        if self.output_file_name == '': 
            logging.info('* The output file name is empty. Default name given is population')
            self.output_file_name = 'population'
        
        # Checks if codebook_fp is a valid path
        valid_codebook_path = self.check_path(file=self.codebook_fp)
        logging.info('* The codebook path provided is not valid.') if not valid_codebook_path else ''    

        # Checks if tract_data is a valid path
        valid_tract_data = self.check_path(file=self.tractdata_fp)
        logging.info('* The tract data path provided is not valid.') if not valid_tract_data else ''    
        
        # Checks if ipums_shp_fp is a valid path
        valid_ipums_shp = self.check_path(file=self.ipums_shp_fp)
        logging.info('* The IPUMS shapefile path provided is not valid.') if not valid_ipums_shp else '' 

        # Output only one time
        valid_inputs = valid_output_file_name and valid_codebook_path and valid_tract_data and valid_ipums_shp

        return valid_inputs
    
    def check_path(self, file=''):
        ''' 
        Checks if file exists at the path specified.
        
        INPUTS:
            - file: the file path to check (optional)
        
        OUTPUTS:
            - Boolean: True if the file exists, False otherwise
        '''
        # If no specific file is given, check the instance's file_path
        if file == '':
            file = self.file_path
        
        # Check if the path is a file
        file_exists = os.path.isfile(file)

        return file_exists
