#!/usr/bin/env python
# coding: utf-8

# # Process single cell profiles

# ## Import libraries

# In[1]:


import argparse
import pathlib
import pprint
import sys

import pandas as pd
from pycytominer import annotate, feature_select, normalize


# ## Papermill parameters

# In[2]:


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

# In[3]:


# Path to directories
if hlhs_run:
    cleaned_dir = pathlib.Path("./data/cleaned_profiles/hlhs")
    output_dir = pathlib.Path("./data/single_cell_profiles/hlhs")
else:
    cleaned_dir = pathlib.Path("./data/cleaned_profiles")
    output_dir = pathlib.Path("./data/single_cell_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# operations to perform for feature selection
feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "variance_threshold",
    "correlation_threshold",
] # run drop na columns first for improved speed

# Extract the plate names from the file name
plate_names = [
    file.stem.replace("_cleaned", "") for file in cleaned_dir.glob("*.parquet")
]


# Filter out plates that already exist in output_dir (any file starting with that plate name)
to_process = []
for plate in plate_names:
    pattern = f"{plate}*.parquet"
    processed = any(output_dir.glob(pattern))
    if not processed:
        to_process.append(plate)

print("Plate names to process:")
pprint.pprint(to_process)


# ## Set dictionary with plates to process

# In[ ]:


# Select the platemap based on the run type
platemap_file = (
    "hlhs_heart_failure_subtypes_platemap.csv"
    if hlhs_run
    else "nf_heart_failure_subtypes_platemap.csv"
)

# Create plate info dictionary
plate_info_dictionary = {
    name: {
        "profile_path": str(
            pathlib.Path(next(iter(cleaned_dir.glob(f"{name}_*.parquet")))).resolve(
                strict=True
            )
        ),
        "platemap_path": pathlib.Path(
            f"../0.download_data/metadata/platemaps/{platemap_file}"
        ).resolve(strict=True),
    }
    for name in to_process
}

# View the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Process data with pycytominer

# In[5]:


for plate, info in plate_info_dictionary.items():
    print(f"Performing pycytominer pipeline for {plate}")

    # Dynamically set output file names based on the suffix
    output_annotated_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_annotated.parquet")
    )
    output_normalized_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_normalized.parquet")
    )
    output_feature_select_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_feature_selected.parquet")
    )

    profile_df = pd.read_parquet(info["profile_path"])
    platemap_df = pd.read_csv(info["platemap_path"])

    # Rename Image_FileName and Image_PathName and BoundingBox columns to keep downstream
    profile_df.rename(
        columns={
            col: (
                col.replace("Image_FileName", "Metadata_Image_FileName").replace(
                    "Image_PathName", "Metadata_Image_PathName"
                )
                if "Image_FileName" in col or "Image_PathName" in col
                else (
                    f"Metadata_{col}"
                    if "BoundingBox" in col and not col.startswith("Metadata_")
                    else col
                )
            )
            for col in profile_df.columns
        },
        inplace=True,
    )

    print("Performing annotation for", plate, "...")
    # Step 1: Annotation
    annotate(
        profiles=profile_df,
        platemap=platemap_df,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
        output_file=output_annotated_file,
        output_type="parquet",
    )

    # Load the annotated parquet file to fix metadata columns names
    annotated_df = pd.read_parquet(output_annotated_file)

    # Rename columns using the rename() function
    column_name_mapping = {
        "Image_Metadata_Site": "Metadata_Site",
        "Image_Metadata_Condition": "Metadata_Condition",
    }

    annotated_df.rename(columns=column_name_mapping, inplace=True)

    # Fix NaN treatment issue in Metadata_treatment column
    annotated_df["Metadata_treatment"] = (
        annotated_df["Metadata_treatment"].replace({None: "None"}).fillna("None")
    )

    # Save the modified DataFrame back to the same location
    annotated_df.to_parquet(output_annotated_file, index=False)

    # Normalize to the None treatments
    samples = "all"

    print(
        "Performing normalization for", plate, "using this samples parameter:", samples
    )

    # Step 2: Normalization
    normalized_df = normalize(
        profiles=output_annotated_file,
        method="mad_robustize",
        output_file=output_normalized_file,
        output_type="parquet",
        samples=samples,
    )

    print("Performing feature selection for", plate, "...")
    # Step 3: Feature selection
    feature_select(
        output_normalized_file,
        operation=feature_select_ops,
        na_cutoff=0,
        blocklist_file="./blocklist_features.txt",
        output_file=output_feature_select_file,
        output_type="parquet",
    )

    # Load back in the feature selected data to drop specific features that leaked in
    feature_selected_df = pd.read_parquet(output_feature_select_file)
    cols_to_drop = [
        col
        for col in feature_selected_df.columns
        if ("Costes" in col or "Location" in col or "Manders" in col or "RWC " in col)
        and not col.startswith("Metadata_")
    ]
    feature_selected_df.drop(columns=cols_to_drop, inplace=True)
    feature_selected_df.to_parquet(output_feature_select_file, index=False)
    print(
        f"Annotation, normalization, and feature selection have been performed for {plate}"
    )


# In[6]:


# Check output file
test_df = pd.read_parquet(output_feature_select_file)

# Test if Costes and Location features were dropped (not including columns that start with Metadata_)
for col in test_df.columns:
    # Skip metadata columns
    if col.startswith("Metadata_"):
        continue
    if "Costes" in col or "Location" in col:
        raise ValueError(
            f"Feature selection failed to drop {col} from the feature selected data."
        )

# Print the number of features (do not have Metadata_* prefix)
non_metadata_features = [
    col for col in test_df.columns if not col.startswith("Metadata_")
]
print(f"Number of features: {len(non_metadata_features)}")

print(test_df.shape)
test_df.head(2)

