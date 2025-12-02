#!/usr/bin/env python
# coding: utf-8

# # Process single cell profiles

# ## Import libraries

# In[1]:


import pathlib
import pprint

import pandas as pd

from pycytominer import annotate, normalize, feature_select


# ## Set paths and variables

# In[2]:


# Set this flag to True for cleaned data (applied QC), or False for no QC applied
use_cleaned_data = True

# Path to directories
converted_dir = pathlib.Path("./data/converted_profiles")
cleaned_dir = pathlib.Path("./data/cleaned_profiles")

# Set the directory based on the flag
data_dir = cleaned_dir if use_cleaned_data else converted_dir

# output path for single-cell profiles
output_dir = pathlib.Path("./data/single_cell_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract the plate names from the file name
plate_names = [
    file.stem.replace("_converted", "") for file in converted_dir.glob("*.parquet")
]
print("Plate names to process:")
pprint.pprint(plate_names)

# operations to perform for feature selection
feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "blocklist",
    "drop_na_columns",
]


# ## Set dictionary with plates to process

# In[3]:


# Create plate info dictionary
plate_info_dictionary = {
    name: {
        "profile_path": str(
            pathlib.Path(list(data_dir.rglob(f"{name}_*.parquet"))[0]).resolve(
                strict=True
            )
        ),
        "platemap_path": str("../0.download_data/metadata/dmso_training_platemap.csv"),
    }
    for name in plate_names
}

# View the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Process data with pycytominer

# In[4]:


# Determine suffix based on use_cleaned_data
suffix = "_no_QC" if not use_cleaned_data else ""

for plate, info in plate_info_dictionary.items():
    print(f"Performing pycytominer pipeline for {plate}")

    # Dynamically set output file names based on the suffix
    output_annotated_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_annotated{suffix}.parquet")
    )
    output_normalized_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_normalized{suffix}.parquet")
    )
    output_feature_select_file = str(
        pathlib.Path(f"{output_dir}/{plate}_sc_feature_selected{suffix}.parquet")
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
    }

    annotated_df.rename(columns=column_name_mapping, inplace=True)

    # Fix NaN treatment issue in Metadata_treatment column
    annotated_df["Metadata_treatment"] = (
        annotated_df["Metadata_treatment"].replace({None: "None"}).fillna("None")
    )

    # Save the modified DataFrame back to the same location
    annotated_df.to_parquet(output_annotated_file, index=False)

    # Normalize to the None treatments
    samples = "Metadata_heart_number == 2 and Metadata_treatment == 'None'"

    print(
        "Performing normalization for", plate, "using this samples parameter:", samples
    )

    # Step 2: Normalization
    normalized_df = normalize(
        profiles=output_annotated_file,
        method="standardize",
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
        output_file=output_feature_select_file,
        output_type="parquet",
    )

    # Load back in the feature selected data to drop specific features that leaked in (Costes and Location features)
    feature_selected_df = pd.read_parquet(output_feature_select_file)
    cols_to_drop = [
        col
        for col in feature_selected_df.columns
        if ("Costes" in col or "Location" in col) and not col.startswith("Metadata_")
    ]
    feature_selected_df.drop(columns=cols_to_drop, inplace=True)
    feature_selected_df.to_parquet(output_feature_select_file, index=False)
    print(
        f"Annotation, normalization, and feature selection have been performed for {plate}"
    )


# In[5]:


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

