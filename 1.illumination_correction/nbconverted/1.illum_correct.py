#!/usr/bin/env python
# coding: utf-8

# # Run illumination correction on data
# 
# Note: We load in the CellProfiler IC pipeline to use for this process.

# ## Import libraries

# In[1]:


import pathlib
import pprint

import sys

sys.path.append("../utils")
import cp_parallel


# ## Set paths and variables

# ### Set the constants

# In[2]:


# set the run type for the parallelization
run_name = "illum_correction"

# set if this run if for the HLHS dataset
hlhs_run = False


# ### Set up paths

# In[3]:


# set main output dir for all plates if it doesn't exist
output_dir = pathlib.Path("./illum_directory")
output_dir.mkdir(exist_ok=True)

# set base directory for where the images are located (WILL NEED TO CHANGE ON YOUR LOCAL MACHINE)
base_dir = pathlib.Path(
    "/home/jenna/mnt/Way_McKinsey_Cardiac_Fibrosis/Heart_Subtypes_data/2_with NF/"
).resolve(strict=True)

# list for plate names based on folders to use to create dictionary
plate_names = []

if hlhs_run:
    # find the plates inside each condition plate
    for parent in ["x1", "x2", "x3"]:
        parent_dir = base_dir / parent

        # Read the plate name from child folder
        plate_names.extend(
            [folder.name for folder in parent_dir.iterdir() if folder.is_dir()]
        )
else:
    # there are likely no conditions in this case, so just read the plate names from the base directory
    plate_names.extend(
        [folder.name for folder in base_dir.iterdir() if folder.is_dir()]
    )

# Sort plate names
plate_names = sorted(plate_names)

print("Found", len(plate_names), "plates:")
for plate in plate_names:
    print(plate)


# ## Create dictionary with all plate data to run CellProfiler in parallel

# In[4]:


# set path to the illum pipeline
path_to_pipeline = pathlib.Path("./pipeline/illum.cppipe").resolve(strict=True)

# set path to loaddata csv files
loaddata_dir = pathlib.Path("./loaddata_csvs").resolve(strict=True)

# create plate info dictionary with all parts of the CellProfiler CLI command to run in parallel
plate_info_dictionary = {
    name: {
        "path_to_loaddata": next(iter(loaddata_dir.rglob(f"loaddata_{name}.csv"))).resolve(
            strict=True
        ),
        "path_to_output": pathlib.Path(f"{output_dir}/{name}/"),
        "path_to_pipeline": path_to_pipeline,
    }
    for name in plate_names
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Run CellProfiler Parallel
# 
# Note: We do not run this code cell as we will run this process through the script.

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=plate_info_dictionary, run_name=run_name, group_level="plate"
)

