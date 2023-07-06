# ECHO-AIR
**E**stimating **C**oncentrations and **H**ealth **O**utcomes - **A**utomated **I**SRM **R**esource 

Author: *Libby Koolik, UC Berkeley*

Last modified July 6, 2023

----
**NOTE**: this repository is currently under construction while migrating to its new home.

----

## Table of Contents
* Purpose and Goals ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#purpose-and-goals))
* Methodology ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#methodology))
* Code Details ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#code-details))
* Running the Tool ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#running-the-tool))

----

## Purpose and Goals
The Intervention Model for Air Pollution (InMAP) is a powerful first step towards lowering key technical barriers by making simplifying assumptions that allow for streamlined predictions of PM<sub>2.5</sub> concentrations resulting from emissions-related policies or interventions.\[[*](https://doi.org/10.1371/journal.pone.0176131)\] InMAP performance has been validated against observational data and WRF-Chem, and has been used to perform source attribution and exposure disparity analyses.\[[*](https://doi.org/10.1126/sciadv.abf4491), [*](https://doi.org/10.1073/pnas.1816102116), [*](https://doi.org/10.1073/pnas.1818859116)\] The InMAP Source-Receptor Matrix (ISRM) was developed by running the full InMAP model tens of thousands of times to understand how a unit perturbation of emissions from each grid cell affects concentrations across the grid. However, both InMAP and the ISRM require considerable computational and math proficiency to run and an understanding of various atmospheric science principles to interpret. Furthermore, estimating health impacts requires additional knowledge and calculations beyond InMAP. Thus, a need arises for a standalone and user-friendly process for comparing air quality health disparities associated with various climate change policy scenarios.

The ultimate goal of this repository is to create a pipeline for estimating disparities in health impacts associated with incremental changes in emissions. Annual average PM<sub>2.5</sub> concentrations are estimated using the [InMAP Source Receptor Matrix](https://www.pnas.org/doi/full/10.1073/pnas.1816102116) for California.

----

## Methodology ##
The ECHO-AIR model works by a series of two modules. First, the model estimates annual average change in PM<sub>2.5</sub> concentrations as part of the **Concentration Module**. Second, the excess mortality resulting from the concentration change is calculated in the **Health Module**.

### Concentration Module Methodology ###
The InMAP Source Receptor Matrix (ISRM) links emissions sources to changes in receptor concentrations. There is a matrix layer for each of the five precursor species: primary PM<sub>2.5</sub>, ammonia (NH<sub>3</sub>), oxides of nitrogen (NOx), oxides of sulfur (SOx), and volatile organic compounds (VOC). By default, the tool uses the California ISRM. For each of these species in the California ISRM, the ISRM matrix dimensions are: 3 elevations by 21,705 sources by 21,705 receptors. The three elevations of release height within the ISRM are:
* Less than 57 meters
* Between 57 and 140 meters
* Greater than 760 meters.

The tool is capable of reading in a different ISRM, if specified by the user. 

The units of each cell within the ISRM are micrograms per meter cubed per microgram per second, or concentration per emissions. 

The concentration module has the following steps. Details about the code handling each step are described in the Code Details([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#code-details)) section below.

1. **Preprocessing**: the tool will load the emissions shapefile and perform a series of formatting checks and adjustments. Any updates will be reported through the command line. Additionally, the ISRM layers will be imported as an object. The tool will also identify how many of the ISRM layers are required for concentration calculations.

For each layer triggered in the preprocessing step: 

2. **Emissions Re-Allocation**: the tool will re-grid emissions to the ISRM grid.
   1. The emissions shape and the ISRM shape are intersected.
   2. Emissions for the intersection object are allocated from the original emissions shape by the percent of the original emissions area that is contained within the intersection.
   3. Emissions are summed by ISRM grid cell.
   4. Note: for point source emissions, a small buffer is added to each point to allocate to ISRM grid cells.
3. **Matrix Multiplication**: Once the emissions are re-gridded to the ISRM grid, they are multiplied by the ISRM grid level for the corresponding layer. 

Once all layers are done:

4. **Sum all Concentrations**: concentrations of PM<sub>2.5</sub> are summed by ISRM grid cell.

### Health Module Methodology ###
The ECHO-AIR health module follows US EPA BenMAP CE methodology and CARB guidance. 

Currently, the tool is only built out to use the Krewski et al. (2009), endpoint parameters and functions.([*](https://www.healtheffects.org/publication/extended-follow-and-spatial-analysis-american-cancer-society-study-linking-particulate)) The Krewski function is as follows:

$$ \Delta M = 1 - ( \frac{1}{\exp(\beta_{d} \times C_{i})} ) \times I_{i,d,g} \times P_{i,g} $$

where $\beta$ is the endpoint parameter from Krewski et al. (2009), $d$ is the disease endpoint, $C$ is the concentration of PM<sub>2.5</sub>, $i$ is the grid cell, $I$ is the baseline incidence, $g$ is the group, and $P$ is the population estimate. The tool takes the following steps to estimate these concentrations.

1. **Preprocessing**: the tool will merge the population and incidence data based on geographic intersections using the `health_data.py` object type. 

2. **Estimation by Endpoint**: the tool will then calculate excess mortality by endpoint:
   1. The population-incidence data are spatially merged with the exposure concentrations estimated in the Concentration Module.
   2. For each row of the intersection, the excess mortality is estimated based on the function of choice (currently, only Krewski).
   3. Excess mortality is summed across age ranges by ISRM grid cell and racial/ethnic group.

Once all endpoints are done:

3. **Export and Visualize**: excess mortality is exported as a shapefile and as a plot.

### Other Features ###
ECHO-AIR has a command called `check-setup` that allows the user to make sure that all of the code and data files are properly saved and named in order to make sure that the program will run.

----

## Code Details ##
Below is a brief table of contents for the Code Details section of the Readme.
* Requirements ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#requirements))
* `isrm_calcs.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#isrm_calcspy))
* Supporting Code ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#supporting-code))
   * `concentration_layer.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#concentration-layerpy))
   * `concentration.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#concentrationpy))
   * `control_file.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#control_filepy))
   * `emissions.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#emissionspy))
   * `health_data.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#health_datapy))
   * `isrm.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#isrmpy))
   * `population.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#populationpy))
* Scripts ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#scripts))
   * `environmental_justice_calcs.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#environmental-justice-calcspy))
   * `health_impact_calcs.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#health-impact-calcspy))
   * `tool_utils.py` ([*](https://github.com/lkoolik/isrm_health_calculations/blob/main/README.md#tool-utilspy))

### Requirements
The code is written in Python 3. The library requirements are included in this repository as `requirements.txt`. For completeness, they are reproduced here:
* attrs==21.4.0
* certifi==2021.10.8
* click==8.1.2
* click-plugins==1.1.1
* cligj==0.7.2
* cycler==0.11.0
* DateTime==4.5
* Fiona==1.8.21
* fonttools==4.32.0
* geopandas==0.10.2
* kiwisolver==1.4.2
* matplotlib==3.5.1
* munch==2.5.0
* numpy==1.22.3
* packaging==21.3
* pandas==1.4.2
* pathlib==1.0.1
* Pillow==9.1.0
* pyarrow==7.0.0
* pyparsing==3.0.8
* pyproj==3.3.0
* python-dateutil==2.8.2
* pytz==2022.1
* Rtree==1.0.0
* scipy==1.8.0
* seaborn==0.11.2
* Shapely==1.8.1.post1
* six==1.16.0
* zope.interface==5.4.0

Python libraries can be installed by running `pip install -r requirements.txt` on a Linux/Mac command line.  

### `isrm_calcs.py`
The `isrm_calcs.py` script is the main script file that drives the tool. This script operates the command line functionality, defines the health impact calculation objects, calls each of the supporting functions, and outputs the desired files. The `isrm_calcs.py` script is not split into functions or objects, instead, it is run through two sections: (1) Initialization and (2) Run Program.

#### Initialization
In the initialization section of `isrm_calcs.py`, the parser object is created in order to interface with the command line. The parser object is created using the `argparse` library. 

Currently, the only arguments accepted by the parser object are `-i` for input file, `-h` for help, and `--check-setup` to run a setup check. 

Once the parser is defined, the control file object is created using `control_file.py` class object. A number of metadata variables are defined from the control file. 

Next, a number of internally saved data file paths are saved.

Finally, the output_region is defined based on the `get_output_region` function defined in `tool_utils.py`. The output region is then stored for use in later functions.

#### Run Program
The run program section of the code is split into two modes. If the CHECK_INPUTS flag is given, the model will run in check mode, where it will check that each of the inputs is valid and then quit. If the CHECK_INPUTS flag is not given, the model will run the full program. 

It will start by creating a log file using the `setup_logging` function. Once the logging is set up, an output directory is created using the `create_output_dir` function from `tool_utils.py`. It will also create a shapefile subdirectory within the output folder directory using `create_shape_out`. The tool will also create an `output_region` geodataframe from user inputs for use in future steps.

Then, the model will begin the concentration module. This starts by defining an emissions object and an isrm object using the `emissions.py` and `isrm.py` supporting class objects. The concentrations will be estimated using the `concentration.py` object, which relies on the `concentration_layer.py` object. The concentrations will then be output as a map of total exposure concentration and a shapefile with detailed exposure information. 

Next, the model will run environmental justice exposure calculations using the `create_exposure_df`, `get_overall_disparity`, and `estimate_exposure_percentile` functions from the `environmental_justice_calcs.py` file. The exposure percentiles will then be plotted and exported using the `plot_percentile_exposure` function. If the control file has indicated that exposure data should be output (using the 'OUTPUT_EXPOSURE' flag), a shapefile of exposure concentrations by population group will be output in the output directory.

Finally, if indicated by the user, the model will begin the health module. It will create the health input object using the `health_data.py` library and then estimate the three endpoints of excess mortality using `calculate_excess_mortality` from the `health_impact_calcs` file. Each endpoint will then be mapped and exported using `visualize_and_export_hia`.

The model utilizes parallel computing to increase efficiency and reduce runtime. As such, many of these steps do not happen exactly in the order presented above. 

The program has completed when a box stating "Success! Run complete." shows on the screen.

#### Check Module
If enabled in the control file, the program will run in `check` mode, which will run a number of checks built into the `emissions`, `isrm`, and `population` objects. Once it runs all checking functions, it will quit and inform the user of the result. 


### Supporting Code
To streamline calculations and increase functionality of the code, python `classes` were created. These class definitions are saved in the `supporting` folder of the repository. The following sections outline how each of these classes work.

#### `concentration_layer.py` 
The `concentration_layer` object runs ISRM-based calculations using a single vertical layer of the ISRM grid. The object inputs an emissions object (from `emissions.py`), the ISRM object (from `isrm.py`), and the layer number corresponding to the vertical layer of the ISRM grid. The object then estimates concentrations at ground-level resulting from emissions at that vertical layer release range.

*Inputs*
* `emis_obj`: the emissions object, as defined by `emissions.py`
* `isrm_obj`: the ISRM object, as defined by `isrm.py`
* `layer`: the layer number (0, 1, or 2)

*Attributes*
* `isrm_id`: a Series of all ISRM grid cell IDs
* `receptor_id`: a Series of all receptor IDs
* `isrm_geom`: the geometry (geographic attributes) of the ISRM grid
* `crs`: the coordinate reference system associated with the ISRM grid
* `name`: a string representing the run name preferred by the user
* `check`: a Boolean indicating whether the program should run, or if it should just check the inputs (useful for debugging)
* `verbose`: a Boolean indicating whether the user wants to run in verbose mode

*Calculated Attributes*
* `PM25e`, `NH3e`, `VOCe`, `NOXe`, `SOXe`: geodataframes of the emissions (for each pollutant) from that layer re-allocated onto the ISRM grid
* `pPM25`, `pNH4`, `pVOC`, `pNO3`, `pSO4`: geodataframes of the concentrations from each primary pollutant from the emissions of that pollutant in that `layer`
* `detailed_conc`: geodataframe containing columns for each primary pollutant's contribution to the total ground-level PM<sub>2.5</sub> concentrations

*Simple Functions*
* `allocate_emissions`: inputs the emissions layer and the ISRM geography, and re-allocates the emissions to the ISRM geography using an area-based allocation procedure
* `cut_emissions`: inputs the pollutant geodataframe from the emissions object and slices it based on the minimum and maximum release heights (minimum inclusive, maximum exclusive) associated with the ISRM vertical layer
* `process_emissions`: for each of the five primary pollutants, runs `cut_emissions` and then `allocate_emissions` to return the geodataframes of emissions of each primary pollutant released in the `layer` allocated to the ISRM grid
* `get_concentration`: for a pollutant's emission layer (`POLe`), the ISRM matrix for that pollutant, and the `layer` ID, estimates the concentration at ground-level for the primary pollutant (`pPOL`)
* `combine_concentrations`: merges together all five of the primary pollutant concentration geodataframes (`pPOL`) and adds them together to get total ground-level concentrations resulting from emissions released in that `layer`

#### `concentration.py` 
The `concentration` object runs ISRM-based calculations for each of the vertical layer's of the ISRM grid by processing individual `concentration_layer` objects. The object inputs an emissions object (from `emissions.py`) and the ISRM object (from `isrm.py`). The object then estimates total concentrations at ground-level resulting from emissions.

*Inputs*
* `emis_obj`: the emissions object, as defined by `emissions.py`
* `isrm_obj`: the ISRM object, as defined by `isrm.py`
* `detailed_conc_flag`: a Boolean indicating whether concentrations should be output at a detailed level or not

*Attributes*
* `isrm_id`: a Series of all ISRM grid cell IDs
* `isrm_geom`: the geometry (geographic attributes) of the ISRM grid
* `crs`: the coordinate reference system associated with the ISRM grid
* `name`: a string representing the run name preferred by the user
* `run_calcs`: a Boolean indicating whether the program should run, or if it should just check the inputs (useful for debugging)
* `verbose`: a Boolean indicating whether the user wants to run in verbose mode

*Calculated Attributes*
* `detailed_conc`: geodataframe of the detailed concentrations at ground-level combined from all three vertical layers
* `detailed_conc_clean`: simplified geodataframe of the detailed concentrations at ground-level combined from all three vertical layers
* `total_conc`: geodataframe with total ground-level PM<sub>2.5</sub> concentrations across the ISRM grid

*Internal Functions*
* `run_layer`: estimates concentrations for a single layer by creating a `concentration_layer` object for that layer
* `combine_concentrations`: checks for each of the layer flags in the `emissions` object, and then calls the `run_layer` function for each layer that is flagged. Then, combines the concentrations from each layer flagged into the three concentration geodataframes described above

*External Functions*
* `visualize_concentrations`: draws a map of concentrations for a variable (`var`) and exports it as a PNG into an output directory (`output_dir`) of choice
* `export_concentrations`: exports concentrations as a shapefile into an output directory (`output_dir`) of choice

#### `control_file.py`
The `control_file` object is used to check and read the control file for a run:

*Inputs*
* `file_path`: the file path of the control file

*Attributes*
* `valid_file`: a Boolean indicating whether or not the control file path is valid
* `keywords`: a hardcoded list of the keywords that should be present in the control file
* `blanks_okay`: a hardcoded list of whether each keyword can be blank (based on order of `keywords`)
* `valid_structure`, `no_incorrect_blanks`: Boolean keywords based on internal checks of the control file format
* `run_name`: a string representing the run name preferred by the user
* `emissions_path`: a string representing the path to the emissions input file
* `emissions_units`: a string representing the units of the emissions data
* `isrm_path`: a string representing the path of the folder storing ISRM numpy layers and geodata
* `population_path`: a string representing the path to the population input file
* `check`: a Boolean indicating whether the program should run, or if it should just check the inputs (useful for debugging)
* `population_path`: a string representing the path to the population data file
* `verbose`: a Boolean indicating whether the user wants to run in verbose mode
* `output_exposure`: a Boolean indicating whether exposure should be output
* `detailed_conc`: a Boolean indicating whether concentrations should should be output as totals or by pollutant

*Internal Functions*
* `check_path`: checks if a file exists at the given control file path
* `get_input_value`: gets the input for a given keyword
* `check_control_file`: runs all of the internal checks to confirm the control file is valid
* `get_all_inputs`: imports all values from the control file
* `get_region_dict`: loads all of the acceptable values for the various regions
* `region_check_helper`: a helper function for checking the region of interest and region category inputs
* `check_inputs`: checks that all inputs are valid once imported

*External Functions*
* `get_file_path`: returns the file path

#### `emissions.py`
The `emissions` object is primarily built off of `geopandas`. It has the following attributes:

*Inputs*
* `file_path`: the file path of the raw emissions data
* `output_dir`: a filepath string for the output directory
* `f_out`: a string containing the filename pattern to be used in output files
* `units`: units associated with the emissions (e.g., μg/s)
* `name`: a plain English name tied to the emissions data, either provided or automatically generated from the filepath
* `details_to_keep`: any additional details to be preserved throughout the processing (e.g., sector, fuel type) (not fully built out yet)
* `filter_dict`: filters the emissions inputs based on inputted dictionary (not fully built out yet)
* `load_file`: a Boolean indicating whether or not the file should be loaded (for debugging)
* `verbose`: a Boolean indicating whether or not detailed logging statements should be printed

*Attributes*
* `valid_file`: a Boolean indicating whether or not the file provided is valid
* `valid_units`: a Boolean indicating whether or not emissions units are compatible with the program
* `valid_emissions`: a Boolean indicating whether or not emissions passed required tests
* `file_type`: the type of file being used to provide raw emissions data (for now, only `.shp` is allowed)
* `geometry`: geospatial information associated with the emissions input
* `crs`: the inherent coordinate reference system associated with the emissions input
* `emissions_data`: complete, detailed emissions data from the source
* `emissions_data_clean`: simplified emissions in each grid cell

*Calculated Attributes*
* `PM25`: primary PM<sub>2.5</sub> emissions in each grid cell
* `NH3`: ammonia emissions in each grid cell
* `VOC`: VOC compound emissions in each grid cell
* `NOX`: NOx emissions in each grid cell
* `SOX`: SOx emissions in each grid cell
* `L0_flag`, `L1_flag`, `L2_flag`, `linear_interp_flag`: Booleans indicating whether each layer should be calculated based on emissions release heights

*Internal Functions*
* `get_file_path`: returns the file path
* `get_name`: returns the name associated with the emissions (`emissions_name`)
* `get_unit_conversions`: returns two dictionaries of built-in unit conversions
* `check_path`: uses the `path` library to check if the provided `file_path` exists and if the file is a file
* `check_units`: checks that the provided units are valid against the `get_unit_conversions` dictionaries
* `load_emissions`: detects the filetype of the emissions file and calls the appropriate load function
* `load_shp`: loads the emissions data from a shapefile
* `load_feather`: loads the emissions data from a feather file
* `load_csv`: loads the emissions data from a csv file
* `check_height`: checks that the height column is present in the emissions file; if not, assumes emissions are released at ground-level
* `check_emissions`: runs a number of checks on the emissions data to ensure data are valid before running anything
* `map_pollutant_names`: replaces pollutant names if they are not found in the emissions data based on near-misses (e.g., PM2.5 for PM25)
* `filter_emissions`: filters the emissions based on the `filter_dict` input
* `check_geo_types`: checks what geometries are present in the emissions shapefile (e.g., points, polygons, multipolygons); if points exist, uses `buffer_emis` to convert to polygons
* `buffer_emis` converts points to polygons by adding a buffer of `dist` 
* `clean_up`: simplifies the emissions data by removing unnecessary dimensions, converting units as appropriate, and updating the column names
* `convert_units`: converts units from provided units to μg/s using the unit dictionaries built-in
* `split_polutants`: converts the emissions layer into separate objects for each pollutant
* `which_layers`: determines the `L0_flag`, `L1_flag`, `L2_flag`, and `linear_interp_flag` variables based on the HEIGHT column of the emissions data

*External Functions*
* `visualize_emissions`: creates a simple map of emissions for a provided pollutant
* `get_pollutant_layer`: pulls a single pollutant layer based on `pol_name`

#### `health_data.py` 
The `health_data` object stores and manipulates built-in health data (population and incidence rates) from BenMAP. It inputs a dictionary of filepaths and two Boolean run options (`verbose` and `race_stratified`) to return dataframes of population, incidence, and combined population-incidence information (`pop_inc`).

*Inputs*
* `pop_alloc`: a geodataframe of population allocated to the ISRM grid geometry
* `incidence_fp`: a string containing the file path to the background incidence dataset
* `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
* `race_stratified`: a Boolean indicating whether race-stratified incidence rates should be used

*Calculated Attributes*
* `population`: a geodataframe containing the population allocated to the ISRM grid geometry
* `incidence`: a geodataframe containing the raw incidence data from BenMAP
* `pop_inc`: a geodataframe containing the combined population and incidence data based on the requested geographies

*Internal Functions*
* `load_data`: reads in the population and incidence data from feather files
* `update_pop`: updates the population dataset by melting (unpivot) and renaming columns
* `update_inc`: updates the incidence dataset by pivoting columns around endpoints and renaming columns
* `get_incidence_lookup`: creates a small incidence lookup table based on the name and age ranges
* `get_incidence_pop`: helper function that returns the incidence for a given name, race, age range, and endpoint
* `make_incidence_lookup`: creates a lookup dictionary using the `get_incidence_pop` function for each endpoint
* `incidence_by_age`: creates a smaller incidence table for merging by calling `get_incidence_lookup` for each endpoint
* `combine_pop_inc`: creates the `pop_inc` dataframe by doing a spatial merge on the population and incidence data and then using lookup tables to determine the appropriate values

#### `isrm.py` 
The `isrm` object loads, stores, and manipulates the ISRM grid data. 

*Inputs*
* `isrm_path`: a string representing the folder containing all ISRM data
* `output_region`: a geodataframe of the region for results to be output, as calculated by `get_output_region` in `tool_utils.py`
* `region_of_interest`: the name of the region contained in the `output_region`
* `load_file`: a Boolean indicating whether or not the file should be loaded (for debugging)
* `verbose`: a Boolean indicating whether or not detailed logging statements should be printed

*Attributes*
* `nh3_path`, `nox_path`, `pm25_path`, `sox_path`, `voc_path`: the filepath strings for each of the primary pollutant ISRM variables
* `valid_file`: a Boolean indicating whether or not the file provided is valid
* `valid_geo_file`: a Boolean indicating whether the ISRM geometry file provided is valid
* `geodata`: a geodataframe containing the ISRM feather file information
* `crs`: the inherent coordinate reference system associated with the ISRM geometry
* `geometry`: geospatial information associated with the ISRM geometry

*Calculated Attributes*
* `receptor_IDs`: the IDs associated with ISRM receptors within the `output_region`
* `receptor_geometry`: the geospatial information associated with the ISRM receptors within the `output_region`
* `PM25`, `NH3`, `NOx`, `SOX`, `VOC`: the ISRM matrices for each of the primary pollutants

*Internal Functions*
* `get_isrm_files`: appends the file names to the isrm_path input to generate full file paths
* `check_path`: checks if the files exist at the paths specified (both data and geo files)
* `load_and_cut`: loads the numpy layers for a pollutant and trims the columns of each vertical layer's matrix to only include the `receptor_IDs` within the `output_region`
* `load_isrm`: calls the `load_and_cut` function for each ISRM numeric layer and returns a list of pollutant matrices
* `load_geodata`: loads the feather file into a geopandas dataframe
* `clip_isrm`: clips the ISRM receptor geodata to only the relevant ones based on the `output_region` (i.e., returns the `receptor_IDs` and `receptor_geometry` objects)

*External Functions*
* `get_pollutant_layer`: returns the ISRM matrix for a single pollutant
* `map_isrm`: simple function for mapping the ISRM grid cells

#### `population.py` 
The `population` object stores detailed Census tract-level population data for the environmental justice exposure calculations and the health impact calculations from an input population dataset.

*Inputs*
* `file_path`: the file path of the raw population data
* `load_file`: a Boolean indicating whether or not the file should be loaded (for debugging)
* `verbose`: a Boolean indicating whether or not detailed logging statements should be printed

*Attributes*
* `valid_file`: a Boolean indicating whether or not the file provided is valid
* `geometry`: geospatial information associated with the emissions input
* `pop_all`: complete, detailed population data from the source
* `pop_geo`: a geodataframe with population IDs and spatial information
* `crs`: the inherent coordinate reference system associated with the emissions input
* `pop_exp`: a geodataframe containing the population information with associated spatial information, summarized across age bins
* `pop_hia`: a geodataframe containing the population information with associated spatial information, broken out by age bin

*Internal Functions*
* `check_path`: checks to see if the file exists at the path specified and returns whether the file is valid
* `load_population`: loads the population data based on the file extension
* `load_shp`: loads the population shapefile data using geopandas and post-processes
* `load_feather`: loads the population feather data using geopandas and post-processes
* `make_pop_exp`: makes the exposure population data frame by summing across age bins
* `make_pop_hia`: makes the health impact assessment population data frame by retaining key information

*External Functions*
* `project_pop`: projects the population data to a new coordinate reference system
* `allocate_population`: reallocates population into new geometry using a spatial intersect

### Scripts
To streamline calculations and increase functionality of the code, Python scripts were created for major calculations/operations. Scripts are saved in the `scripts` folder of the repository. The following sections outline the contents of each script file, and how the functions inside them work.

#### `environmental_justice_calcs.py` 
The `environmental_justice_calcs` script file contains a number of functions that help calculate exposure metrics for environmental justice analyses.

1. `create_exposure_df`: creates a dataframe ready for exposure calculations
   1. Inputs:
      * `conc`: concentration object from `concentration.py`
      * `isrm_pop_alloc`: population object (from `population.py`) re-allocated to the ISRM grid cell geometry
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
   2. Outputs
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group
   3. Methodology:
      1. Pulls the total concentration from the concentration object
      2. Grabs the population by racial/ethnic group from the population object
      3. Merges the concentration and population data based on the ISRM ID
      4. Adds the population weighted mean exposure as a column of the geodataframe using `add_pwm_col`
2. `add_pwm_col`: adds an intermediate column that multiplies population by exposure concentration
   1. Inputs:
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group
      * `group`: the racial/ethnic group name
   2. Outputs:
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group, now with PWM column
   3. Methodology:
      1. Creates a column called `group`+'_PWM'.
      2. Multiplies exposure concentration by `group` population
      3. Returns the new dataframe
   4. Important Notes:
      * The new column is not actually a population-weighted mean, it is just an intermediate for calculating PWM in the next step.
3. `get_pwm`: estimates the population-weighted mean exposure for a given group
   1. Inputs:
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group
      * `group`: the racial/ethnic group name
   2. Outputs:
      * `PWM_group`: the group-level population weighted mean exposure concentration (float)
   3. Methodology:
      1. Creates a variable for the group PWM column (as created in `add_pwm_col`
      2. Estimates PWM by adding across the `group`_PWM column and dividing by the total `group` population
4. `get_overall_disparity`: returns a table of overall disparity metrics by racial/ethnic group
   1. Inputs:
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group
   2. Outputs:
      * `pwm_df`: a dataframe containing the PWM, absolute disparity, and relative disparity of each group
   3. Methodology:
      1. Creates an empty dataframe with the groups as rows
      2. Estimates the group population weighted mean using the `get_pwm` function
      3. Estimates the absolute disparity as `Group_PWM` - `Total_PWM`
      4. Estimates the relative disparity as the `Absolute Disparity`/`Total_PWM`
5. `estimate_exposure_percentile`: creates a dataframe of exposure percentiles for plotting
   1. Inputs:
      * `exposure_gdf`: a geodataframe with the exposure concentrations and allocated population by racial group
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
   2. Outputs:
      * `df_pctl`: a dataframe of exposure concentrations by percentile of population exposed by group
   3. Methodology:
      1. Creates a copy of the `exposure_gdf` dataframe to prevent writing over the original.
      2. Sorts the dataframe by PM2.5 concentration and resets the index.
      3. Iterates through each racial/ethnic group, performing the following:
         1. Creates a small slice of the dataframe that is only the exposure concentration and the `group`.
         2. Estimates the cumulative sum of population in the sorted dataframe.
         3. Estimates the total population of the `group`.
         4. Estimates percentile as the population in the grid cell divided by the total population of the `group`.
         5. Adds the percentile column into the main dataframe.
6. `run_exposure_calcs`: calls the other exposure justice functions in order
   1. Inputs:
      * `conc`: concentration object from `concentration.py`
      * `isrm_pop_alloc`: population object (from `population.py`) re-allocated to the ISRM grid cell geometry
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
   2. Outputs:
      * `exposure_gdf`: a dataframe containing the exposure concentrations and population estimates for each group
      * `exposure_pctl`: a dataframe of exposure concentrations by percentile of population exposed by group
      * `exposure_disparity`: a dataframe containing the PWM, absolute disparity, and relative disparity of each group
   3. Methodology:
      1. Calls the `create_exposure_df` function.
      2. Calls the `get_overall_disparity` function.
      3. Calls the `estimate_exposure_percentile` function.
7. `export_exposure_gdf`: exports the exposure concentrations and population estimates as a shapefile
   1. Inputs: 
      * `exposure_gdf`: a dataframe containing the exposure concentrations and population estimates for each group
      * `shape_out`: a filepath string of the location of the shapefile output directory
      * `f_out`: the name of the file output category (will append additional information)
   2. Outputs:
      * A shapefile will be output into the `shape_out` directory.
      * The function returns `fname` as a surrogate for completion (otherwise irrelevant)
   3. Methodology:
      1. Creates a filename and path for the export.
      2. Updates the columns slightly for shapefile naming
      3. Exports the shapefile.
8. `export_exposure_csv`: exports the exposure concentrations and population estimates as a CSV file
   1. Inputs: 
      * `exposure_gdf`: a dataframe containing the exposure concentrations and population estimates for each group
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information)
   2. Outputs:
      * A CSV file will be output into the `output_dir`.
      * The function returns `fname` as a surrogate for completion (otherwise irrelevant)
   3. Methodology:
      1. Creates a filename and path for the export.
      2. Updates the column names for more straightforward interpretation
      3. Exports the results as a comma-separated value (CSV) file.
9. `export_exposure_disparity`: exports the exposure concentrations and population estimates as a shapefile
   1. Inputs: 
      * `exposure_disparity`: a dataframe containing the population-weighted mean exposure concentrations for each group
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information)
   2. Outputs:
      * A shapefile will be output into the `output_dir`.
      * The function returns `fname` as a surrogate for completion (otherwise irrelevant)
   3. Methodology:
      1. Creates a filename and path for the export.
      2. Updates the columns and values slightly for more straightforward interpretation
      3. Exports the results as a comma-separated value (CSV) file.
10. `plot_percentile_exposure`: creates a plot of exposure concentration by percentile of each group's population
   1. Inputs: 
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information)
      * `exposure_pctl`: a dataframe of exposure concentrations by percentile of population exposed by group
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
   2. Outputs:
      * The function does not return anything, but a lineplot image (PNG) will be output into the `output_dir`.
   3. Methodology:
      1. Creates a melted (un-pivoted) version of the percentiles dataframe.
      2. Multiplies the percentile by 100 to span 0-100 instead of 0-1.
      3. Maps the racial/ethnic group names to better formatted names (e.g., "HISLA" --> "Hispanic/Latino")
      4. Draws the figure using the `seaborn` library's `lineplot` function.
      5. Saves the file as `f_out` + '_PM25_Exposure_Percentiles.png' into the `out_dir`.
11. `export_exposure`: calls each of the exposure output functions in parallel
   1. Inputs: 
      * `exposure_gdf`: a dataframe containing the exposure concentrations and population estimates for each group
      * `exposure_disparity`: a dataframe containing the population-weighted mean exposure concentrations for each group
      * `exposure_pctl`: a dataframe of exposure concentrations by percentile of population exposed by group
      * `shape_out`: a filepath string of the location of the shapefile output directory
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information)
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed 
   2. Outputs:
      * The function does not return anything, but a shapefile will be output into the `output_dir`.
   3. Methodology:
      1. Creates a filename and path for the export.
      2. Updates the columns slightly for shapefile naming
      3. Exports the shapefile.
12. `create_rename_dict`: makes a global rename code dictionary for easier updating
   1. Inputs: None
   2. Outputs:
      * `logging_code`: a dictionary that maps endpoint names to log statement codes
   3. Methodology:
      1. Defines a dictionary and returns it.

#### `health_impact_calcs.py` 
The `health_impact_calcs` script file contains a number of functions that help calculate health impacts from exposure concentrations.

1. `create_hia_inputs`: creates the hia_inputs object.
   1. Inputs:
      * `pop`: population object input
      * `load_file`: a Boolean telling the program to load or not
      * `verbose`: a Boolean telling the program to return additional log statements or not
      * `geodata`: the geographic data from the ISRM
      * `incidence_fp`: a string containing the filepath where the incidence data is stored
   2. Outputs:
      * a health data object ready for health calculations
   3. Methodology
      1. Allocates population to the ISRM grid using the population object and the ISRM geodata.
      2. Initializes a health_data object from that allocated population.

2. `krewski`: defines a Python function around the Krewski et al. (2009) function and endpoints
   1. Inputs:
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed
      * `conc`: a float with the exposure concentration for a given geography
      * `inc`: a float with the background incidence for a given group in a given geography
      * `pop`: a float with the population estimate for a given group in a given geography
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
   2. Outputs
      * a float estimating the number of excess mortalities for the `endpoint` across the group in a given geography
   3. Methodology:
      1. Based on the `endpoint`, grabs a `beta` parameter from Krewski et al. (2009).
      2. Estimates excess mortality using the following equation, where $\beta$ is the endpoint parameter from Krewski et al. (2009), $d$ is the disease endpoint, $C$ is the concentration of PM<sub>2.5</sub>, $i$ is the grid cell, $I$ is the baseline incidence, $g$ is the group, and $P$ is the population estimate.

$$ 1 - ( \frac{1}{\exp(\beta_{d} \times C_{i})} ) \times I_{i,d,g} \times P_{i,g} $$

3. `create_logging_code`: makes a global logging code for easier updating
   1. Inputs: None
   2. Outputs:
      * `logging_code`: a dictionary that maps endpoint names to log statement codes
   4. Methodology:
      1. Defines a dictionary and returns it.
     
4. `calculate_excess_mortality`: estimates excess mortality for a given `endpoint` and `function`
   1. Inputs:
      * `conc`: a float with the exposure concentration for a given geography
      * `health_data_obj`: a `health_data` object as defined in the `health_data.py` supporting script
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `function`: the health impact function of choice (currently only `krewski` is built out)
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed      
   2. Outputs
      * `pop_inc_conc`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
   3. Methodology:
      1. Creates clean, simplified copies of the `detailed_conc` method of the `conc` object and the `pop_inc` method of the `health_data_obj`.
      2. Merges these two dataframes on the ISRM_ID field.
      3. Estimates excess mortality on a row-by-row basis using the `function`.
      4. Pivots the dataframe to get the individual races as columns.
      5. Adds the geometry back in to make it geodata.
      6. Updates the column names such that the excess mortality columns are ENDPOINT_GROUP.
      7. Merges the population back into the dataframe.
      8. Cleans up the dataframe.
 
5. `plot_total_mortality`: creates a map image (PNG) of the excess mortality associated with an `endpoint` for a given `group`.
   1. Inputs:
      * `hia_df`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
      * `ca_shp_fp`: a filepath string of the California state boundary shapefile
      * `group`: the racial/ethnic group name
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information) 
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed      
   2. Outputs
      * `fname`: a string filename made by combining the `f_out` with the `group` and `endpoint`.
   3. Methodology:
      1. Sets a few formatting standards within `seaborn` and `matplotlib.pyplot`.
      2. Creates the output file directory and name string using `f_out`, `group`, and `endpoint`.
      3. Reads in the California boundary and projects the `hia_df` to match the coordinate reference system of the California dataset.
      4. Clips the dataframe to the California boundary.
      5. Adds area-normalized columns to the `hia_df` for more intuitive plotting.
      6. Grabs the minimums and sets them to 10<sup>-9</sup> in order to avoid logarithm conversion errors.
      7. Updates the 'MORT_OVER_POP' column to avoid 100% mortality that arises from the update in step 6.
      8. Initializes the figure and plots four panes:
         1. Population density: plots the area-normalized population estimates for the group on a log-normal scale.
         2. PM<sub>2.5</sub> exposure concentrations: plots the exposure concentration on a log-normal scale.
         3. Excess mortality per area: plots the excess mortality per unit area on a log-normal scale.
         4. Excess mortality per population: plots the excess mortality per population for the group on a log-normal scale.
      9. Performs a bit of clean-up and formatting before exporting.
  
6. `export_health_impacts`: exports mortality as a shapefile
   1. Inputs:
      * `hia_df`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
      * `group`: the racial/ethnic group name
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information) 
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed      
   2. Outputs
      * `fname`: a string filename made by combining the `f_out` with the `group` and `endpoint`.
   3. Methodology:
      1. Creates the output file path (`fname`) using inputs.
      2. Creates endpoint short labels and updates column names since shapefiles can only have ten characters in column names.
      3. Exports the geodataframe to shapefile.

7. `export_health_impacts_csv`: exports mortality as a csv
   1. Inputs:
      * `hia_df`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information) 
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed      
   2. Outputs
      * `fname`: a string filename made by combining the `f_out` with the `group` and `endpoint`.
   3. Methodology:
      1. Creates the output file path (`fname`) using inputs.
      2. Revises column names for clarity
      3. Exports the geodataframe to csv.

8. `create_summary_hia`: creates a summary table of health impacts by racial/ethnic group
   1. Inputs:
      * `hia_df`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed   
      * `l`: an intermediate string that has the endpoint label string (e.g., ACM_)
      * `endpoint_nice`: an intermediate string that has a nicely formatted version of the endpoint (e.g., All Cause)    
   2. Outputs
      * `hia_summary`: a summary dataframe containing population, excess mortality, and excess mortality rate per demographic group
   3. Methodology:
      1. Cleans up the hia_df by changing column names and splitting population and mortality
      2. Gets total population and mortality by group
      3. Combines into one dataframe and cleans it up for export

9. `visualize_and_export_hia`: calls `plot_total_mortality` and `export_health_impacts` in one clean function call.
   1. Inputs:
      * `hia_df`: a dataframe containing excess mortality for the `endpoint` using the `function` provided
      * `ca_shp_fp`: a filepath string of the California state boundary shapefile
      * `group`: the racial/ethnic group name
      * `endpoint`: a string containing either 'ALL CAUSE', 'ISCHEMIC HEART DISEASE', or 'LUNG CANCER'
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information) 
      * `shape_out`: a filepath string for shapefiles
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed      
   2. Outputs
      * `hia_summary`: a summary dataframe containing population, excess mortality, and excess mortality rate per demographic group
   3. Methodology:
      1. Calls `plot_total_mortality`.
      2. Calls `export_health_impacts.

10. `combine_hia_summaries`: combines the three endpoint summary tables into one export file
   1. Inputs:
      * `acm_summary`: a summary dataframe containing population, excess all-cause mortality, and all-cause mortality rates
      * `ihd_summary`: a summary dataframe containing population, excess IHD mortality, and IHD mortality rates 
      * `lcm_summary`: a summary dataframe containing population, excess lung cancer mortality, and lung cancer mortality rates
      * `output_dir`: a filepath string of the location of the output directory
      * `f_out`: the name of the file output category (will append additional information) 
      * `verbose`: a Boolean indicating whether or not detailed logging statements should be printed 
   2. Outputs: None
   3. Methodology:
      1. Merges the summary dataframes together
      2. Removes excess columns
      3. Saves as CSV file
      
11. `create_rename_dict`: makes a global rename code dictionary for easier updating
   1. Inputs: None
   2. Outputs:
      * `logging_code`: a dictionary that maps endpoint names to log statement codes
   3. Methodology:
      1. Defines a dictionary and returns it.

#### `tool_utils.py` 
The `tool_utils` library contains a handful of scripts that are useful for code execution.

1. `check_setup`: checks that the isrm_health_calculations local clone is set up properly
   1. Inputs: None
   2. Outputs:
      * `valid_setup`: a Boolean indicating if the setup is valid or not
   3. Methodology:
      1. Gets the programs current working directory.
      2. Checks that all the script and supporting files exist where they are supposed to.
      3. Checks that all key data files are saved where they should (not including ISRM)
      4. Checks that the CA_ISRM is located in the data folder with all necessary objects, but does not consider this an improper setup, as the user may have their own ISRM.
      5. Reports any missing files or directories.

2. `setup_logging`: sets up the log file capability using the `logging` library
   1. Inputs:
      * `debug_mode`: a Boolean indicating if log statements should be returned in debug mode or not
   2. Outputs:
      * `tmp_logger`: a filepath string associated with a temporary log file that will be moved as soon as the output directory is created
   3. Methodology:
      1. Defines useful variables for the `logging` library.
      2. Creates a temporary log file path (`tmp_logger`) that allows the file to be created before the output directory.
      3. Suppresses all other library warnings and information.
      4. Sets the formatting system for log statements.

3. `verboseprint`: sets up the verbose printing mechanism for global usage
   1. Inputs:
      * `verbose`: a Boolean indicating if it is in verbose mode or not
      * `text`: a string to be returned if the program is in verbose mode
   2. Outputs: None
   3. Methodology:
      1. Checks if verbose is True.
      2. If True, creates a log statement.
      3. If False, does nothing. 

4. `report_version`: reports the current working version of the tool
   1. Inputs: None
   2. Outputs: None
   3. Methodology: adds statements to the log file about the tool version

5. `create_output_dir`: creates the output directory for saving files
   1. Inputs:
      * `batch`: the batch name 
      * `name`: the run name
   2. Outputs:
      * `output_dir`: a filepath string for the output directory
      * `f_out`: a string containing the filename pattern to be used in output files
   3. Methodology:
      1. Grabs the current working directory of the tool and defines 'outputs' as the sub-directory to use.
      2. Checks to see if the directory already exists. If it does exists, automatically increments by 1 to create a unique directory.
      3. Creates `f_out` by removing the 'out' before the `output_dir`.
      4. Creates the output directory.

6. `create_shape_out`: creates the output directory for saving shapefiles
   1. Inputs:
      * `output_dir`: a filepath string for the output directory
   2. Outputs:
      * `shape_out`: a filepath string for the shapefile output directory
   3. Methodology:
      1. Creates a directory within the `output_dir` called 'shapes'.
      2. Stores this name as `shape_out`.

7. `get_output_region`: creates the output region geodataframe
   1. Inputs:
      * `region_of_interest`:  the name of the region to be contained in the `output_region`
      * `region_category`: a string containing the region category for the output region, must be one of 'AB','AD', or 'C' for Air Basins, Air Districts, and Counties
      * `output_geometry_fps`: a dictionary containing a mapping between `region_category` and the filepaths
      * `ca_fps`: a filepath string containing the link to the California border shapefile
   2. Outputs
      * `output_region`: a geodataframe containing only the region of interest
   3. Methodology:
      1. Checks if the `region_of_interest` is California, in which case, it just reads in the California shapefile.
      2. If California is not the `region_of_interest`:
         1. Gets the filepath of the output region based on the `region_category` from the `output_geometry_fps` dictionary.
         2. Reads in the file as a geodataframe.
         3. Clips the geodataframe to the `region_of_interest`.

----

## Running the Model

The tool is configured to be run on a [Mac](https://lkoolik.github.io/isrm_tool/) or via Linux terminal (including Windows Subsystem for Linux) on the [Google Cloud or Windows Subsystem for Linux](https://docs.google.com/document/d/1aurYIaGMi6BCvQaK6cEyrb5amSAX8TXTYiB2ko2N8FU/). Instructions for each of those are linked in the previous sentence.

----

## Acknowledgments

In alphabetical order, the following people are acknowledged for their support and contributions:
* Dr. Álvaro Alvarado (OEHHA): advisor to development
* Dr. Joshua Apte (UC Berkeley): project PI
* Amy Budahn (OEHHA, now CARB): advisor to development
* Thomas Le (UC Berkeley): investigated pipelines for regulatory data
* Dr. Julian Marshall (UW): advisor to development
* Dr. Laurel Plummer (OEHHA): advisor to development
* Justin Ward (Google): supported parallelization of main program
