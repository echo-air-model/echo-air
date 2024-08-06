#!/isrm-env/bin/python python3
# -*- coding: utf-8 -*-
"""
Main Run File for Creating Population File

@author: amyryao
Last updated: 2024-06-18
"""
#%% Import useful libraries, supporting objects, and scripts
# Useful libraries for main script
from pathlib import Path
import sys
sys.path.insert(0,'./supporting')
sys.path.insert(0,'./scripts')
import argparse
import logging
import os
import time
import datetime
import shutil
import concurrent.futures
import platform
from inspect import currentframe, getframeinfo

# Import supporting objects
from pop_control_file import pop_control_file
from census import census

# Import supporting scripts

from tool_utils import *

parser = argparse.ArgumentParser(description="Runs the ECHO-AIR Population Pre-Processor. For more information, see the documentation site: https://github.com/echo-air-model/echo-air-model.github.io/blob/population_documentation/docs/additional_information/creating_population_files.md") # I will update the link when the documentation is pushed! 

# Add necessary arguments
parser.add_argument("-i", "--inputs", help="control file path", type=str)
parser.add_argument("--debug", help="enable for debugging mode", action="store_true")

# Parse all arguments
args = parser.parse_args()
debug_mode = args.debug

#% Create the log file and update logging configuration
tmp_logger = setup_logging(debug_mode) 

# Read control file and create useful variables
cf = pop_control_file(args.inputs)

# Load all the control file info
file_name = cf.output_file_name
codebook_fp = cf.codebook_fp
tractdata_fp = cf.tractdata_fp
ipums_shp_fp = cf.ipums_shp_fp
verbose = cf.verbose

# Similar to create_output_dir, this creates an output directory for files generated and returns a file name for additional file outputs
def create_pop_output_dir(batch):
    ''' 
    Creates the output directory for files generated with 'pop' in the name.
    
    INPUTS:
        - batch: the batch name 
        
    OUTPUTS:
        - output_dir: a filepath string for the output directory
        - f_out: a string containing the filename pattern to be used in output files
    
    '''
    # Grab current working directory and the 'outputs' sub folder
    parent = os.getcwd()
    sub = 'outputs'
    
    # Output subdirectory will be named with 'pop' and batch
    outdir = f'out_pop_{batch}'
    
    # If the directory already exists, add an integer to the end
    path_exists_flag = path.exists(os.path.join(parent, sub, outdir))
    # Use while loop in case there are multiple directories already made
    while path_exists_flag:
        if outdir == f'out_pop_{batch}':
            n = 0
            outdir_tmp = outdir + '_'        
        else:
            # Need to pull just the last two numbers 
            try:
                n = int(outdir.split('_')[-1]) # Grab the number at the end
                outdir_tmp = outdir[:-2] # Get a temporary output directory
                if outdir_tmp[-1] != '_': # Only one run exists in the directory, need to start at 01
                    n = 0
                    outdir_tmp = outdir_tmp + '_'
            except ValueError: # This means there was no number at the end, so start with 0
                outdir_tmp = outdir + '_'
                n = 0
        # Update to the next n
        next_n = str(n+1).zfill(2)
        outdir = outdir_tmp + next_n
        
        # Check if this path exists
        path_exists_flag = path.exists(os.path.join(parent, sub, outdir))
        
    # Add a new variable f_out that adds more information to output file names
    f_out = outdir[4:] # Cuts off the 'out_' from the start
    
    # Make the directory if it does not already exist
    os.mkdir(os.path.join(parent, sub, outdir))
    output_dir = os.path.join(parent, sub, outdir)
    
    # Print a statement to tell user where to look for files
    logging.info("\n << Output files created will be saved in the following directory: " + output_dir + " >>")
    
    return output_dir, f_out

# Create an output population directory
output_dir, f_out = create_pop_output_dir(file_name)

# Prints to show the start of the preprocessing script
logging.info('\n')
logging.info('╔═══════════════════════════════════════════╗')
logging.info('║ ECHO-AIR Population Pre-Processing Script ║')
logging.info('║ Version 0.0.1                             ║')
logging.info('╚═══════════════════════════════════════════╝')
logging.info('\n')

# Create census object
census_obj = census(codebook_fp, tractdata_fp, ipums_shp_fp, output_dir, f_out, verbose, debug_mode=debug_mode)
# Processes the data and exports the finished population feather folder
census_obj.preprocess_data()

logging.info('\n')
logging.info('╓────────────────────────────────╖')
logging.info('║ Success! Script complete.      ║')
logging.info('╙────────────────────────────────╜\n')
logging.info('\n')
logging.info('<< ECHO-AIR has created and exported all control files indicated. >>')
