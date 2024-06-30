#!/isrm-env/bin/python python3
# -*- coding: utf-8 -*-
"""
Main Run File for Creating Population File

@author: libbykoolik
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

parser = argparse.ArgumentParser(description="Runs ECHO-AIR: an ISRM-based model for estimating PM2.5 concentrations and associated health impacts.")

# Add necessary arguments
parser.add_argument("-i", "--inputs", help="control file path", type=str)
parser.add_argument("--debug", help="enable for debugging mode", action="store_true")
parser.add_argument("--check-setup", help="checks to see if your package is properly set up", action="store_true")
parser.add_argument("-p", "--parallel", help="runs the tool with parallelization", action="store_true")

# Parse all arguments
args = parser.parse_args()
check_setup_flag = args.check_setup
debug_mode = args.debug
run_parallel = args.parallel

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

# Create the output directory
output_dir, f_out = create_output_dir(file_name, '')

logging.info('╓─────────────────────────────────╖')
logging.info('║ Creating Population Input Files ║')
logging.info('╙─────────────────────────────────╜')
logging.info('\n')

# Create census object
census_obj = census(codebook_fp, tractdata_fp, ipums_shp_fp, output_dir, f_out, verbose, debug_mode=debug_mode)
# Processes the data and exports the finished population feather folder
census_obj.preprocess_data()

logging.info('\n')
logging.info('╓────────────────────────────────╖')
logging.info('║ Success! Script complete.      ║')
logging.info('╙────────────────────────────────╜\n')
logging.info('<< ECHO-AIR has created and exported all control files indicated. >>')
