#!/usr/bin/env python
# coding: utf-8

# # Convert SQLite output to parquet file with CytoTable

# ## Import libraries

# In[1]:


import argparse
import pathlib
import sys

import pandas as pd

# cytotable will merge objects from SQLite file into single cells and save as parquet file
from cytotable import convert, presets

import logging

# Set the logging level to a higher level to avoid outputting unnecessary errors from config file in convert function
logging.getLogger().setLevel(logging.ERROR)


# ## Papermill parameters

# In[ ]:


# Set if this run is for the HLHS dataset (will be updated by papermill)
hlhs_run = False


# Optional CLI fallback for direct script-style execution.
def _str_to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--hlhs-run", dest="hlhs_run", type=_str_to_bool, default=hlhs_run
)
args, _ = parser.parse_known_args(sys.argv[1:])
hlhs_run = args.hlhs_run


# ## Set paths and variables

# In[2]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"

if hlhs_run:
    # update preset to include site metadata and cell counts
    joins = presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"].replace(
        "Image_Metadata_Well,",
        "Image_Metadata_Well, Image_Metadata_Site, Image_Metadata_Condition,",
    )
    # Add the PathName columns separately
    joins = joins.replace(
        "COLUMNS('Image_FileName_.*'),",
        "COLUMNS('Image_FileName_.*'),\n COLUMNS('Image_PathName_.*'),",
    )
else:
    # update preset to include site metadata and cell counts
    joins = presets.config["cellprofiler_sqlite_pycytominer"]["CONFIG_JOINS"].replace(
        "Image_Metadata_Well,",
        "Image_Metadata_Well, Image_Metadata_Site,",
    )

    # Add the PathName columns separately
    joins = joins.replace(
        "COLUMNS('Image_FileName_.*'),",
        "COLUMNS('Image_FileName_.*'),\n COLUMNS('Image_PathName_.*'),",
    )

# type of file output
dest_datatype = "parquet"

# set path to directory with SQLite files and output directory for processed data
if hlhs_run:
    sqlite_dir = pathlib.Path("../2.extract_features/cp_output/hlhs_run")
    output_dir = pathlib.Path("data/converted_profiles/hlhs")
else:
    sqlite_dir = pathlib.Path("../2.extract_features/cp_output")
    output_dir = pathlib.Path("data/converted_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

plate_names = []

# Select plate name folders if starts with CARD
for file_path in sqlite_dir.iterdir():
    if file_path.is_dir() and file_path.name.startswith("CARD"):
        plate_names.append(file_path.stem)

# print the plate names and how many plates there are (confirmation)
print(f"There are {len(plate_names)} plates in this dataset. Below are the names:")
for name in plate_names:
    print(name)


# ## Convert SQLite to parquet files

# In[3]:


for plate_name in plate_names:
    file_path = sqlite_dir / plate_name
    output_path = pathlib.Path(f"{output_dir}/{file_path.stem}_converted.parquet")
    print("Starting conversion with cytotable for plate:", file_path.stem)
    # Merge single cells and output as parquet file
    convert(
        source_path=str(file_path),
        dest_path=str(output_path),
        dest_datatype=dest_datatype,
        preset=preset,
        joins=joins,
        chunk_size=5000,
    )

print("All plates have been converted with cytotable!")


# # Load in converted profiles to update

# In[4]:


for file_path in output_dir.iterdir():
    # Skip anything that isn't a parquet file directly in this directory (e.g. subfolders)
    if not (file_path.is_file() and file_path.suffix == ".parquet"):
        continue

    # Load the DataFrame from the Parquet file
    df = pd.read_parquet(file_path)

    # If any, drop rows where "Metadata_ImageNumber" is NaN (artifact of cytotable)
    df = df.dropna(subset=["Metadata_ImageNumber"])

    # Rearrange columns and add "Metadata" prefix in one line
    df = df[
        [
            "Nuclei_Location_Center_X",
            "Nuclei_Location_Center_Y",
            "Cells_Location_Center_X",
            "Cells_Location_Center_Y",

        ]
        + [
            col
            for col in df.columns
            if col
            not in [
                "Nuclei_Location_Center_X",
                "Nuclei_Location_Center_Y",
                "Cells_Location_Center_X",
                "Cells_Location_Center_Y",
            ]
        ]
    ].rename(
        columns=lambda col: (
            "Metadata_" + col
            if col
            in [
                "Nuclei_Location_Center_X",
                "Nuclei_Location_Center_Y",
                "Cells_Location_Center_X",
                "Cells_Location_Center_Y",
            ]
            else col
        )
    )

    # Save the processed DataFrame as Parquet in the same path
    df.to_parquet(file_path, index=False)


# ## Check output to confirm process worked
# 
# To confirm the number of single cells is correct, please use any database browser software to see if the number of rows in the "Per_Cells" compartment matches the number of rows in the data frame.

# In[5]:


print(df.shape)
df.head()

