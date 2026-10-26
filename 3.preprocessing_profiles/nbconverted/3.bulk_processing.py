#!/usr/bin/env python
# coding: utf-8

# # Process bulk (well-level) profiles
# 
# We aggregate the QC'd single-cell profiles to the well-level (median) and then annotate, normalize, and feature select the well-level profiles with pycytominer.
# The single-cell profiles are aggregated from the annotated profiles (before single-cell normalization), so the well-level profiles are normalized on their own.
# These bulk profiles are used for the well-level correlation heatmaps in `4.EDA`.

# ## Import libraries

# In[ ]:


import argparse
import pathlib
import pprint
import sys

import pandas as pd
from pycytominer import aggregate, annotate, feature_select, normalize
from pycytominer.cyto_utils import infer_cp_features


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

# In[ ]:


# Path to directories
if hlhs_run:
    single_cell_dir = pathlib.Path("./data/single_cell_profiles/hlhs")
    output_dir = pathlib.Path("./data/bulk_profiles/hlhs")
else:
    single_cell_dir = pathlib.Path("./data/single_cell_profiles")
    output_dir = pathlib.Path("./data/bulk_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# Select the platemap based on the run type
platemap_file = (
    "hlhs_heart_failure_subtypes_platemap.csv"
    if hlhs_run
    else "nf_heart_failure_subtypes_platemap.csv"
)
platemap_path = pathlib.Path(
    f"../0.download_data/metadata/platemaps/{platemap_file}"
).resolve(strict=True)

# Columns to aggregate by (one profile per well)
strata = ["Metadata_Plate", "Metadata_Well"]

# Features that leak through the blocklist in the single-cell processing, so they are not aggregated
features_to_exclude = ["Costes", "Location", "Manders", "RWC "]

# operations to perform for feature selection
# drop_outliers removes features with extreme normalized values (e.g., a feature with a MAD of 0 across
# wells, like the median number of adjacent neighbors, which would otherwise dominate the correlations)
feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "variance_threshold",
    "correlation_threshold",
    "drop_outliers",
] # run drop na columns first for improved speed

# Find the annotated single-cell profiles (QC applied) for each plate
plate_info_dictionary = {
    path.name.replace("_sc_annotated.parquet", ""): path.resolve(strict=True)
    for path in sorted(single_cell_dir.glob("*_sc_annotated.parquet"))
}
if not plate_info_dictionary:
    raise FileNotFoundError(f"No annotated single-cell profiles found in {single_cell_dir}")

pprint.pprint(plate_info_dictionary, indent=4)


# ## Load the platemap
# 
# Only `NA` is read as missing so the `None` treatment is kept as a string (the same as in the single-cell profiles), and healthy hearts have a missing heart failure type.

# In[ ]:


platemap_df = pd.read_csv(platemap_path, keep_default_na=False, na_values=["NA"])
platemap_df.head()


# ## Process data with pycytominer
# 
# For each plate, we:
# 
# 1. Aggregate the single cells per well (median) and count the cells per well
# 2. Annotate the wells with the platemap
# 3. Normalize the wells to all wells on the plate
# 4. Feature select (including dropping features with extreme normalized values)

# In[ ]:


for plate, sc_annotated_path in plate_info_dictionary.items():
    print(f"Performing pycytominer pipeline for {plate}")

    # Dynamically set output file names based on the suffix
    output_aggregated_file = str(output_dir / f"{plate}_bulk_aggregated.parquet")
    output_annotated_file = str(output_dir / f"{plate}_bulk_annotated.parquet")
    output_normalized_file = str(output_dir / f"{plate}_bulk_normalized.parquet")
    output_feature_select_file = str(
        output_dir / f"{plate}_bulk_feature_selected.parquet"
    )

    sc_df = pd.read_parquet(sc_annotated_path)

    # Only aggregate the CellProfiler features that are not excluded
    features = [
        feature
        for feature in infer_cp_features(sc_df)
        if not any(exclude in feature for exclude in features_to_exclude)
    ]

    print("Performing aggregation for", plate, "...")
    # Step 1: Aggregation (median per well)
    aggregate(
        population_df=sc_df,
        strata=strata,
        features=features,
        operation="median",
        compute_object_count=True,
        object_feature="Metadata_Cells_Number_Object_Number",
        output_file=output_aggregated_file,
        output_type="parquet",
    )

    print("Performing annotation for", plate, "...")
    # Step 2: Annotation
    annotate(
        profiles=output_aggregated_file,
        platemap=platemap_df,
        join_on=["Metadata_well_position", "Metadata_Well"],
        output_file=output_annotated_file,
        output_type="parquet",
    )

    print("Performing normalization for", plate, "...")
    # Step 3: Normalization (to all wells on the plate)
    normalize(
        profiles=output_annotated_file,
        method="mad_robustize",
        samples="all",
        output_file=output_normalized_file,
        output_type="parquet",
    )

    print("Performing feature selection for", plate, "...")
    # Step 4: Feature selection
    feature_select(
        profiles=output_normalized_file,
        operation=feature_select_ops,
        na_cutoff=0,
        outlier_cutoff=500,
        blocklist_file="./blocklist_features.txt",
        output_file=output_feature_select_file,
        output_type="parquet",
    )
    print(
        f"Aggregation, annotation, normalization, and feature selection have been performed for {plate}"
    )


# ## Check the output files

# In[ ]:


bulk_summary = []
for plate in plate_info_dictionary:
    for step in ["aggregated", "annotated", "normalized", "feature_selected"]:
        bulk_df = pd.read_parquet(output_dir / f"{plate}_bulk_{step}.parquet")
        features = infer_cp_features(bulk_df)

        # The excluded features should not be in any of the bulk profiles
        leaked = [
            feature
            for feature in features
            if any(exclude in feature for exclude in features_to_exclude)
        ]
        assert not leaked, f"Excluded features found in {plate} {step}: {leaked}"

        # The feature selected profiles should not have extreme values that dominate the correlations
        if step == "feature_selected":
            assert bulk_df[features].abs().max().max() < 500, f"Extreme values found in {plate}"

        bulk_summary.append(
            {
                "plate": plate,
                "step": step,
                "n_wells": bulk_df.shape[0],
                "n_features": len(features),
                "median_cells_per_well": bulk_df["Metadata_Object_Count"].median(),
            }
        )

pd.DataFrame(bulk_summary)


# In[ ]:


# Check the metadata of the final well-level profiles for the last plate
bulk_df.loc[:, bulk_df.columns.str.startswith("Metadata_")].head()

