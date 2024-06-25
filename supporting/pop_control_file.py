#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population Control File Reading Object

@author: amyryao

last modified: 2024-06-26

"""

# Import Libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
import sys
import re
import difflib
import logging

#%%
class pop_control_file:
    '''
    Defines a new object for reading the control file and storing information.
    
    INPUTS:
        - file_path: File path to the control file     
    '''
    def __init__(self, file_path):
        ''' Initializes the control file object'''   
        logging.info('<< Reading Control File >>')
        
        # Initialize path and check that it is valid
        self.file_path = file_path
            
        self.output_file_name, self.codebook_fp, self.tractdata_fp, self.ipums_shp_fp, self.verbose = self.get_all_inputs()
    
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
