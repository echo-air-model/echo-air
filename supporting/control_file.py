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
import os
from os import path
import sys
import re
import difflib
import logging

#%%
class control_file:
    '''
    Defines a new object for reading the control file and storing information.
    
    INPUTS:
        - file_path: File path to the control file 
        
    ATTRIBUTES:
        - valid_file: a Boolean indicating whether or not the control file path is valid
        - keywords: a hardcoded list of the keywords that should be present in the control file
        - blanks_okay: a hardcoded list of whether each keyword can be blank 
          (based on order of `keywords`)
        - valid_structure: Boolean keyword based on internal checks of the control file format
        - no_incorrect_blanks: Boolean keyword based on internal checks of the control file format
        - run_name: a string representing the run name preferred by the user
        - emissions_path: a string representing the path to the emissions input file
        - emissions_units: a string representing the units of the emissions data
        - isrm_path: a string representing the path of the folder storing ISRM numpy layers and geodata
        - population_path: a string representing the path to the population data file
        - check: a Boolean indicating whether the program should run, or if it should just 
          check the inputs (useful for debugging)
        - verbose: a Boolean indicating whether the user wants to run in verbose mode
        - output_exposure: a Boolean indicating whether exposure should be output
        - detailed_conc: a Boolean indicating whether concentrations should should be output as totals or
          by pollutant
        - output_emis: a Boolean indicating whether allocated emissions should be output
        
    '''
    def __init__(self, file_path):
        ''' Initializes the control file object'''   
        logging.info('<< Reading Control File >>')
        
        # Initialize path and check that it is valid
        self.file_path = file_path
        self.valid_file = self.check_path()
        
        # Hardcode the current keywords for simplicity
        self.keywords = ['BATCH_NAME', 'RUN_NAME','EMISSIONS_FILENAME',
                         'EMISSIONS_UNITS', 'POPULATION_FILENAME', 'RUN_HEALTH', 
                         'RACE_STRATIFIED_INCIDENCE', 'CHECK_INPUTS','VERBOSE',
                         'REGION_OF_INTEREST','REGION_CATEGORY','OUTPUT_RESOLUTION',
                         'OUTPUT_EXPOSURE', 'DETAILED_CONC', 'OUTPUT_EMIS',
                          'EMISSIONS_CHANGE', 'BOUNDARY']
        self.blanks_okay = [True, True, False, 
                            False, False, True, 
                            True, True, True,
                            True, True, True,
                            True, True, True, 
                            True, True]
        
        # Run basic checks on control file
        if self.valid_file:
            self.valid_structure, self.no_incorrect_blanks = self.check_control_file()
        else:
            self.valid_structure, self.no_incorrect_blanks = 'NA'
            
        # If checks are good, import values
        if self.valid_structure and self.no_incorrect_blanks and self.valid_file:
            self.batch_name, self.run_name, self.emissions_path, self.emissions_units, self.isrm_path, self.population_path, self.run_health, self.race_stratified, self.check, self.verbose, self.region_of_interest, self.region_category, self.output_resolution, self.output_exposure, self.detailed_conc, self.output_emis, self.emis_delta, self.boundary_change = self.get_all_inputs()
            self.valid_inputs = self.check_inputs()
            if self.valid_inputs:
                logging.info('\n << Control file was successfully imported and inputs are correct >>')
                self.ready = True
            else:
                logging.info('\n << ERROR: Control file was successfully imported but inputs are not correct >>')
                self.ready = False
        else:
            if not self.valid_structure:
                logging.info('\n * Control file did not have the correct structure.')
                logging.info('* Please confirm that the control file has the following keywords exactly once each:')
                logging.info('  * '+'\n   * '.join(self.keywords))
            if not self.no_incorrect_blanks:
                logging.info('* Some keywords were left blank incorrectly in the control file.')
                logging.info('* Only the following keywords can be left blank:')
                logging.info('  * '+'\n   * '.join(pd.Series(self.keywords)[self.blanks_okay].tolist()))
            if not self.valid_file:
                logging.info('* The control file path is not valid.')
            
            logging.info('\n << ERROR: Control File was not correct. Please revise errors and try again. >>\n')
            self.ready = False
            
    def check_path(self, file='', isrm=False):
        ''' Checks if file exists at the path specified '''
        # Use the os library to check the path and the file
        if file == '':
            path_exists = path.exists(self.file_path)
            if isrm==False:
                file_exists = path.isfile(self.file_path)
            else:
                missing = []
                for f in ['ISRM_NH3.npy', 'ISRM_NOX.npy', 'ISRM_PM25.npy','ISRM_SOX.npy', 'ISRM_VOC.npy', 'isrm_geo.feather']:
                    if not(path.isfile(path.join(self.isrm_path, f))):
                        missing.append(f)
                        logging.info('* Issue finding {} in the provided ISRM directory'.format(f))
                file_exists = len(missing) == 0
        else:
            path_exists = path.exists(file)
            if isrm==False:
                file_exists = path.isfile(file)
            else:
                missing = []
                for f in ['ISRM_NH3.npy', 'ISRM_NOX.npy', 'ISRM_PM25.npy','ISRM_SOX.npy', 'ISRM_VOC.npy', 'isrm_geo.feather']:
                    if not(path.isfile(path.join(self.isrm_path, f))):
                        missing.append(f)
                        logging.info('* Issue finding {} in the provided ISRM directory'.format(f))
                file_exists = len(missing) == 0
        return path_exists and file_exists
  
    def get_input_value(self, keyword, upper=False):
        '''Gets the input for the given keyword from a file'''
        
        # Open the file and read through it line by line
        with open(self.file_path, 'r') as file:
            for line in file:
                # Create the exact format of the keyword you're searching for
                re_k = f'- {keyword}:'
                
                # Check if the line contains the keyword
                if re_k in line:
                    # Extract the value following the keyword
                    line_val = line.split(':', 1)[1].strip()
                    
                    # If upper is True, convert the value to uppercase
                    if upper:
                        line_val = line_val.upper()
                        
                    return line_val  # Return the found value immediately
        
        # Return None or raise an error if the keyword is not found
        return None
    
    def check_control_file(self):
        ''' Runs a number of checks to make sure that control file is valid  '''
        ## (TEST 1) Check for correct keywords
        # Create a dictionary that logs whether or not each keyword exists and 
        # how many times
        check_dict = {}
        
        # Loop through the text file and capture keyword usage
        for line in open(self.file_path):
            for k in self.keywords:
                re_k = '- '+k+':' # Grabs exact formatting
                if re_k in line and k not in check_dict.keys(): # Found keyword, doesn't exist in dictionary
                    check_dict[k] = 1
                elif k in line and k in check_dict.keys(): # Found keyword, already exists in dictionary
                    check_dict[k] += 1
                else:
                    pass # not found, keep looping
                    
        # confirm that keywords all present
        all_keywords = set(self.keywords) == set(check_dict.keys())
        
        # confirm that all values are 1
        correct_count = sum(check_dict.values()) == len(self.keywords)
        
        # Combine for test #1 output
        valid_structure = all_keywords & correct_count
        
        if not valid_structure:
            return valid_structure, None
        else:
            ## (TEST 2) Check for blank inputs
            incorrect_blanks = 0 # holder for incorrect blanks
            for line in open(self.file_path):
                for k in zip(self.keywords, self.blanks_okay):
                    if k[1]: # If blanks are okay, ignore
                        pass
                    else:
                        line_val = self.get_input_value(k[0])
                        if line_val == '': # Blanks will report as ''
                            incorrect_blanks += 1 # Add to holder
            no_incorrect_blanks = incorrect_blanks == 0

        return valid_structure, no_incorrect_blanks
    
    def create_emis_delta_dict(self, emis_delta):
        ''' Filters through emis_delta and creates a dictionary '''
        # Defining an empty dict 
        emis_delta_dict = {}

        # Use regular expressions to find all pollutant-value pairs
        matches = re.findall(r'([A-Z0-9]+):\s*([+-]?\d+)', emis_delta)
       
        # Iterate over the matches and add them to the dictionary
        for match in matches:
            pollutant = match[0]  # The pollutant name (e.g., 'PM25')
            value = int(match[1])  # The value (e.g., +30 or -20)
            emis_delta_dict[pollutant] = value

        return emis_delta_dict
    
    def get_all_inputs(self):
        ''' Once it passes the basic control file checks, import the values '''
        mapper = {'Y':True, 'N':False}
        
        # Get each input individually
        batch_name = self.get_input_value('BATCH_NAME')
        run_name = self.get_input_value('RUN_NAME')
        emissions_path = self.get_input_value('EMISSIONS_FILENAME')
        emissions_units = self.get_input_value('EMISSIONS_UNITS')
        isrm_path = self.get_input_value('ISRM_FOLDER')
        population_path = self.get_input_value('POPULATION_FILENAME')
        run_health = self.get_input_value('RUN_HEALTH', upper=True)
        race_stratified = self.get_input_value('RACE_STRATIFIED_INCIDENCE', upper=True)
        region_of_interest = self.get_input_value('REGION_OF_INTEREST', upper=True)
        region_category = self.get_input_value('REGION_CATEGORY', upper=True)
        output_resolution = self.get_input_value('OUTPUT_RESOLUTION', upper=True)
        output_exposure = self.get_input_value('OUTPUT_EXPOSURE', upper=True)
        detailed_conc = self.get_input_value('DETAILED_CONC', upper=True)
        output_emis = self.get_input_value('OUTPUT_EMIS', upper=True)
        emis_delta = self.get_input_value('EMISSIONS_CHANGE', upper=True)
        boundary_change = self.get_input_value('BOUNDARY', upper=True)

        # For ISRM folder, assume CA ISRM if no value is given
        if isrm_path == '':
            logging.info('* No value provided for the ISRM path. Assuming the California ISRM as default.')
            isrm_path = './data/CA_ISRM'
        
        # For HEALTH RUN CONTROLS, assume something if no value is given
        if run_health == '':
            logging.info('* No value provided for the RUN_HEALTH field. Assuming a full run.')
            run_health = True
        else:
            run_health = mapper[run_health] # convert Y/N to True/False
        
        if race_stratified == '' and run_health == True:
            logging.info('* No value provided for the RACE_STRATIFIED field. Assuming non-race stratified incidence rates.')
            race_stratified = False
        elif race_stratified == '' and run_health == False:
            race_stratified = False
        else:
            race_stratified = mapper[race_stratified] # convert Y/N to True/Fals
        
        # For CHECK_INPUTS and VERBOSE, assume something if no value is given
        check = self.get_input_value('CHECK_INPUTS')
        if check == '':
            logging.info('* No value provided for the CHECK_INPUTS field. Assuming a full run.')
            check = False
        else:
            check = mapper[check] # convert Y/N to True/False
        verbose = self.get_input_value('VERBOSE')
        if verbose == '':
            logging.info('* No value provided for the VERBOSE field. Assuming a verbose run.')
            verbose = True
        else:
            verbose = mapper[verbose] # convert Y/N to True/False
            
        # For OUTPUT OPTIONS, assume something if no value is given
        if region_of_interest == '':
            logging.info('* No value provided for the REGION_OF_INTEREST field. Assuming full California run.')
            region_of_interest = 'CA'
        if region_category == '':
            logging.info('* No value provided for the REGION_CATEGORY field. Assuming full California run.')
            region_category = 'STATE'
        if output_resolution == '':
            logging.info('* No value provided for the OUTPUT_RESOLUTION field. Assuming ISRM grid cells.')
            output_resolution = 'ISRM'
            
        # for OUTPUT_EXPOSURE, DETAILED_CONC, and OUTPUT_EMIS check if blank, otherwise map the Y/N
        if output_exposure == '':
            logging.info('* No value provided for the OUTPUT_EXPOSURE field. Assuming output is desired.')
            output_exposure = True
        else:
            output_exposure = mapper[output_exposure]
        if detailed_conc == '':
            logging.info('* No value provided for the DETAILED_CONC field. The tool will output summary concentrations.')
            detailed_conc = False
        else:
            detailed_conc = mapper[detailed_conc]
        if output_emis == '':
            logging.info('* No value provided for the OUTPUT_EMIS field. The tool will not output emissions allocated to the ISRM.')
            output_emis = False
        else:
            output_emis = mapper[output_emis]
        
        # Boundary requires change in emissions. Assume no boundary if nothing specified 
        if emis_delta == '':
            if boundary_change == '': 
                logging.info('* No value provided EMISSIONS_CHANGE field. Assuming same emissions.') 
            else: 
                logging.info('* Was given a boundary but no emissions change. Assuming same emissions.') 
            emis_delta = False
        else: 
            self.emis_delta_dict = self.create_emis_delta_dict(emis_delta)
            if boundary_change == '': 
                logging.info('* No value was provided for BOUNDARY field. Assuming emissions change for entire region.')
    
        return batch_name, run_name, emissions_path, emissions_units, isrm_path, population_path, run_health, race_stratified, check, verbose, region_of_interest, region_category, output_resolution, output_exposure, detailed_conc, output_emis, emis_delta, boundary_change
    
    def get_region_dict(self):
        ''' Hard-coded dictionary of acceptable values for regions '''
        # Define lists and a dictionary of acceptable inputs
        air_basins = ['GREAT BASIN VALLEYS','LAKE COUNTY','LAKE TAHOE','MOJAVE DESERT',
                      'MOUNTAIN COUNTIES','NORTH CENTRAL COAST','NORTH COAST',
                      'NORTHEAST PLATEAU','SACRAMENTO VALLEY','SALTON SEA','SAN DIEGO COUNTY',
                      'SAN FRANCISCO BAY','SAN JOAQUIN VALLEY','SOUTH CENTRAL COAST','SOUTH COAST']
        
        air_districts = ['AMADOR','ANTELOPE VALLEY','BAY AREA','BUTTE','CALAVERAS','COLUSA',
                         'EL DORADO','FEATHER RIVER','GLENN','GREAT BASIN UNIFIED','IMPERIAL',
                         'KERN','LAKE','LASSEN','MARIPOSA','MENDOCINO','MODOC','MOJAVE DESERT',
                         'MONTEREY BAY UNIFIED','NORTH COAST UNIFIED','NORTHERN SIERRA',
                         'NORTHERN SONOMA','PLACER','SACRAMENTO METRO','SAN DIEGO',
                         'SAN JOAQUIN VALLEY UNIFIED','SAN LUIS OBISPO','SANTA BARBARA','SHASTA',
                         'SISKIYOU','SOUTH COAST','TEHAMA','TUOLUMNE','VENTURA','YOLO-SOLANO']
        
        counties = ['ALAMEDA','ALPINE','AMADOR','BUTTE','CALAVERAS','COLUSA','CONTRA COSTA',
                    'DEL NORTE','EL DORADO','FRESNO','GLENN','HUMBOLDT','IMPERIAL','INYO',
                    'KERN','KINGS','LAKE','LASSEN','LOS ANGELES','MADERA','MARIN','MARIPOSA',
                    'MENDOCINO','MERCED','MODOC','MONO','MONTEREY','NAPA','NEVADA','ORANGE',
                    'PLACER','PLUMAS','RIVERSIDE','SACRAMENTO','SAN BENITO','SAN BERNARDINO',
                    'SAN DIEGO','SAN FRANCISCO','SAN JOAQUIN','SAN LUIS OBISPO','SAN MATEO',
                    'SANTA BARBARA','SANTA CLARA','SANTA CRUZ','SHASTA','SIERRA','SISKIYOU',
                    'SOLANO','SONOMA','STANISLAUS','SUTTER','TEHAMA','TRINITY','TULARE',
                    'TUOLUMNE','VENTURA','YOLO','YUBA']
        
        dacs = counties.copy() # DACs are listed by county, and the names are the same -- ignore for now
        
        region_dict = {'AB':air_basins, 'AD':air_districts, 'C':counties}#, 'DAC':dacs}
        
        return region_dict
    
    def region_check_helper(self):
        ''' Simple helper function for checking the region of interest and region_category inputs '''
        # Import the region dict
        region_dict = self.get_region_dict()
        
        ## First check that the REGION_CATEGORY is okay
        region_category = self.region_category
        
        # Define some possible but acceptable wrong names
        possible_wrong_names_dict = {'AB':['AIR BASIN', 'AIRBASIN', 'AIR_BASIN', 'BASIN'],
                                     'AD':['AIR DISTRICT', 'AIRDISTRICT', 'AIR_DISTRICT', 'DISTRICT'],
                                     'C':['COUNTY', 'COUNTIES']}#,
                                     #'DAC':['DISADVANTAGED COMMUNITY', 'EJ COMMUNITY']} # to add DACs in future update
        
        # Check if region_category is correct
        if region_category not in region_dict.keys() and region_category != 'STATE':
            # Set possible replacement to NA as a holder value and create flag variable
            possible_replacement = 'NA'
            region_category_flag = False
            
            # Loop through the dictionary
            for region in possible_wrong_names_dict.keys():
                if region_category in possible_wrong_names_dict[region]: # If it's in the list, update possible_replacement
                    possible_replacement = region
                    region_category_flag = True
                    logging.info('* Incorrect value provided for the REGION_CATEGORY field, however a close replacement ({}) was found.'.format(possible_replacement))
            
            # Update region_category
            region_category = possible_replacement
            
        else:
            region_category_flag = True
                
        if region_category_flag and region_category != 'STATE':
            ## Second, check the region of interest
            region_of_interest = self.region_of_interest
            
            # Get a list of the options
            region_opts = region_dict[region_category]
            
            # Check if region_of_interest is in region_opts
            if region_of_interest not in region_opts and region_of_interest != 'CA':
                # Set possible replacement to NA as a holder value and create flag variable
                possible_replacement = 'NA'
                region_of_interest_flag = False
                
                # Use the difflib function get_close_matches to find the closest match
                closest_match = difflib.get_close_matches(region_of_interest, region_opts, n=1, cutoff=0.6)
                
                try: # If the list does not come back empty, should be able to grab the first entry 
                    closest_match = closest_match[0]
                    logging.info('* Incorrect value provided for the REGION_OF_INTEREST field, however a close replacement ({}) was found.'.format(closest_match))
                    region_of_interest_flag = True
                    region_of_interest = closest_match
                    
                except: # Do nothing, will return region_of_interest = False
                    pass
                    
            else:
                region_of_interest_flag = True
                
            ## Update the self variables for region_category and region_of_interest
            self.region_category = region_category
            self.region_of_interest = region_of_interest
        
        elif region_category_flag and region_category == 'STATE':
            region_of_interest_flag = True
        
        else:
            region_of_interest_flag = False
        
        return region_category_flag, region_of_interest_flag
    
    def out_res_check_helper(self):
        ''' Simple helper function for checking the output resolution input '''
        # Define the output resolutions
        valid_output_resolutions = ['ISRM', 'C', 'AB', 'AD']# to add DACs in future update, 'DACS']
        
        # Check if the input is in the list of valid inputs
        valid_input = self.output_resolution in valid_output_resolutions
        
        # Need to compare against region of interest, define these
        region_category = self.region_category
        permissable_combos = {'AB': ['AD','C','ISRM'],
                              'AD': ['ISRM'],
                              'C': ['ISRM'],
                              'STATE': ['AD','AB','C','ISRM']}        
        
        # Compare the output resolution and region category
        try:
            valid_resolution = self.output_resolution in permissable_combos[self.region_category]
        except:
            valid_resolution = False
        
        # Define valid_output_resolution
        valid_output_resolution = valid_input and valid_resolution
        
        return valid_output_resolution, valid_output_resolutions      
    
    def check_inputs(self):
        ''' Once the inputs are imported, check them '''
        ## (1) Check the batch name and run name are strings
        valid_batch_name = type(self.batch_name) == str
        logging.info('* The batch name provided is not valid (must be string).') if not valid_batch_name else ''
        
        valid_run_name = type(self.run_name) == str
        logging.info('* The run name provided is not valid (must be string).') if not valid_run_name else ''
        
        ## If batch and run name are blank, replace with 'isrm_calcs'
        if self.batch_name == '' and self.run_name == '':
            self.batch_name = 'isrm_calcs'
        
        ## Check the emissions path and units
        # Path checks:
        valid_emissions_path = self.check_path(file=self.emissions_path)
        logging.info('* The emissions path provided is not valid.') if not valid_emissions_path else ''
        
        # Units checks:
        mass_units = ['ug','g','lb','ton','mt','kg'] # mass units from emissions.py
        time_units = ['s','min','hr','day','yr'] # time units from emissions.py

        # Try to split with the division sign
        tmp_units = self.emissions_units.lower().split('/')

        valid_emissions_units = tmp_units[0] in mass_units and tmp_units[1] in time_units
        logging.info('* The emissions units provided is not valid. Acceptable mass emissions units are '+', '.join(mass_units)\
              +' and acceptable time units are '+', '.join(time_units)) if not valid_emissions_units else ''
        
        ## Check the ISRM path
        valid_isrm_path = self.check_path(file=self.isrm_path, isrm=True)
        logging.info('* The ISRM path provided is not valid.') if not valid_isrm_path else ''    
        
        ## Check the population path
        valid_population_path = self.check_path(file=self.population_path)
        logging.info('* The population path provided is not valid.') if not valid_population_path else ''
            
        ## Check the HEALTH RUN CONTROLS
        valid_run_health = type(self.run_health) == bool
        logging.info('* The run health option provided is not valid. Use Y or N or leave blank.') if not valid_run_health else ''
        valid_inc_choice = type(self.race_stratified) == bool
        logging.info('* The race stratified incidence choice provided is not valid. Use Y or N or leave blank.') if not valid_inc_choice else ''
            
        ## Check the RUN CONTROLS
        valid_check = type(self.check) == bool
        logging.info('* The check flag provided is not valid. Use Y or N or leave blank.') if not valid_check else ''
        valid_verbose = type(self.verbose) == bool
        logging.info('* The verbose flag provided is not valid. Use Y or N or leave blank.') if not valid_verbose else ''
        
        ## Check the region and region_of_interest
        valid_region_category, valid_region_of_interest = self.region_check_helper()
        logging.info('* The region category provided is not valid. Valid options include: AB, AD, and C. For fully state, leave blank.') if not valid_region_category else ''
        logging.info('* The region of interest provided is not valid or cannot be validated. Consult user manual for correct options.') if not valid_region_of_interest else ''
        
        ## Check the output_resolution variable
        valid_output_resolution, valid_output_resolutions = self.out_res_check_helper()
        logging.info('* The output resolution provided is not valid. Valid options include: '+', '.join(valid_output_resolutions)) if not valid_output_resolution else ''
        
        ## Check the output_exposure variable
        valid_output_exp= type(self.output_exposure) == bool
        logging.info('* The OUTPUT_EXPOSURE provided is not valid. Use Y or N or leave blank.') if not valid_output_exp else ''
        
        ## Check the detailed_conc variable
        valid_detailed_conc= type(self.detailed_conc) == bool
        logging.info('* The DETAILED_CONC provided is not valid. Use Y or N or leave blank.') if not valid_detailed_conc else ''
        
        ## Check the output_emis variable
        valid_output_emis = type(self.output_emis) == bool
        logging.info('* The DETAILED_CONC provided is not valid. Use Y or N or leave blank.') if not valid_output_emis else ''
        
        ## Check the emis_delta variable
        valid_emis_delta = True

        # Regex patterns for - numbers
        pattern = re.compile(r'^-?\d+$|^0$')

        if self.emis_delta:
            for pol, value in self.emis_delta_dict.items():
                if value is None:
                    logging.info(f"* The EMISSIONS_CHANGE value for {pol} is None and not valid.")
                    valid_emis_delta = False
                    continue  
                
                # Ensure the value is a string for regex matching
                value_str = str(value)

                # Check if the string value matches the regex pattern
                if not pattern.match(value_str):
                    logging.info(f"* The EMISSIONS_CHANGE provided is not valid for {pol}. Use a number with a '-' or a standalone '0'.")
                    valid_emis_delta = False
        
        # Assign the change in emissions as the newly created dictionary
        if valid_emis_delta: 
            self.emis_delta = self.emis_delta_dict
        
        ## Check the boundary path 
        if self.boundary_change == '':
            valid_boundary_path = True # If there is no boundary, continue to proceed
        else: 
            valid_boundary_path = self.check_path(file=self.boundary_change)
            logging.info('* The boundary path for emissions change provided is not valid.') if not valid_boundary_path else ''
            
        ## Output only one time
        valid_inputs = valid_batch_name and valid_run_name and valid_emissions_path and \
            valid_emissions_units and valid_isrm_path and valid_population_path and valid_run_health and \
                valid_inc_choice and valid_check and valid_verbose and valid_region_category and \
                    valid_region_of_interest and valid_output_resolution and valid_output_exp and valid_detailed_conc and \
                        valid_output_emis and valid_emis_delta and valid_boundary_path

        return valid_inputs