#!/usr/bin/env python
# coding: utf-8

# # Perform single-cell quality control
# 
# In this notebook, we perform single-cell quality control using coSMicQC. We use features from the AreaShape and Intensity modules to assess the quality of the segmented single-cells.

# ## Import modules

# In[1]:


import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cosmicqc import find_outliers
from cytodataframe import CytoDataFrame


# ## Papermill parameters

# In[ ]:


# Set default plate name (will be updated by papermill)
# Available plates for testing (copy/paste one below):
#   CARD-CelIns-CX7_260814100001
#   CARD-CelIns-CX7_260817090001
#   CARD-CelIns-CX7_260817180001
plate_name = "CARD-CelIns-CX7_260817180001"

# Set if this run is for the HLHS dataset (will be updated by papermill)
hlhs_run = False

# Set to False to skip rendering CytoDataFrame cell images. Rendering is useful for
# interactive QC review, but can be slow or hang during full, non-interactive papermill
# runs across many plates.
render_images = True


# In[ ]:


# Optional CLI fallback for direct script-style execution.
def _str_to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--plate", "--plate-name", dest="plate_name", default=plate_name)
parser.add_argument(
    "--hlhs-run", dest="hlhs_run", type=_str_to_bool, default=hlhs_run
)
parser.add_argument(
    "--render-images", dest="render_images", type=_str_to_bool, default=render_images
)
args, _ = parser.parse_known_args(sys.argv[1:])
plate_name = args.plate_name
hlhs_run = args.hlhs_run
render_images = args.render_images


# ## Set paths and variables

# In[ ]:


# Set data directories
if hlhs_run:
    data_dir = pathlib.Path("./data/converted_profiles/hlhs")
    cleaned_dir = pathlib.Path("./data/cleaned_profiles/hlhs")
    qc_summary_dir = pathlib.Path("./data/hlhs")
else:
    data_dir = pathlib.Path("./data/converted_profiles")
    cleaned_dir = pathlib.Path("./data/cleaned_profiles")
    qc_summary_dir = pathlib.Path("./data")
cleaned_dir.mkdir(parents=True, exist_ok=True)
qc_summary_dir.mkdir(parents=True, exist_ok=True)

# Set outline context directory
if hlhs_run:
    outline_context_dir = pathlib.Path(
        f"../2.extract_features/cp_output/hlhs_run/{plate_name}"
    )
else:
    outline_context_dir = pathlib.Path(f"../2.extract_features/cp_output/{plate_name}")

# Directory to save qc figures
qc_fig_dir = pathlib.Path("./qc_figures")
qc_fig_dir.mkdir(exist_ok=True)


# ## Load in plate to perform QC on

# In[4]:


if not plate_name:
    available_plates = sorted(
        path.stem.replace("_converted", "")
        for path in data_dir.glob("*_converted.parquet")
    )
    raise ValueError(
        "Set plate_name with papermill using `-p plate_name <plate>`. "
        f"Available plates: {available_plates}"
    )

plate = plate_name
file_path = data_dir / f"{plate}_converted.parquet"

if not file_path.exists():
    available_plates = sorted(
        path.stem.replace("_converted", "")
        for path in data_dir.glob("*_converted.parquet")
    )
    raise FileNotFoundError(
        f"No converted parquet found for plate_name={plate!r}: {file_path}. "
        f"Available plates: {available_plates}"
    )

# Load in converted plate data
plate_df = pd.read_parquet(file_path)

# Add plate metadata column if absent
if "Image_Metadata_Plate" not in plate_df.columns:
    plate_df["Image_Metadata_Plate"] = plate

print(plate_df.shape)
plate_df.head()


# In[5]:


# set compartment for segmentation mask
compartment = "Nuclei"

# channels to include for cytodataframe visualization
channels = ["OrigDNA", "OrigActin"]

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    *[f"Metadata_{compartment}_Location_Center_{axis}" for axis in ("X", "Y")],
    *[f"Image_FileName_{ch}" for ch in channels],
    *[f"Image_PathName_{ch}" for ch in channels],
    *[
        f"{compartment}_AreaShape_BoundingBox{bound}_{axis}"
        for bound in ("Maximum", "Minimum")
        for axis in ("X", "Y")
    ],
]


# In[6]:


# create an outline and orig mapping dictionary to map original images to outlines
# note: we turn off formatting here to avoid the key-value pairing definition
# from being reformatted by black, which is normally preferred.
# fmt: off
outline_to_orig_mapping = {}
for record in plate_df[
    [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]
].to_dict(orient="records"):
    site_raw = str(record["Image_Metadata_Site"])
    # extract digits from the site string (e.g., 'f09' -> '09')
    site_digits = re.sub(r"\D", "", site_raw)
    if site_digits == "":
        site_fmt = site_raw
    else:
        site_fmt = f"{int(site_digits):02d}"

    key = rf"{compartment}Outlines_{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}_{record['Image_Metadata_Site']}.tiff"
    value = rf"{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}f{site_fmt}d\d+.TIF"
    outline_to_orig_mapping[key] = value
# fmt: on

next(iter(outline_to_orig_mapping.items()))


# ## Oversegmented nuclei

# In[ ]:


if plate_name == "CARD-CelIns-CX7_260407120001":
    # find large nuclei clusters
    feature_thresholds = {
        "Nuclei_Intensity_MassDisplacement_DNA": 2.0,
        "Nuclei_AreaShape_Compactness": 2.0,
    }
else:
    # find large nuclei clusters
    feature_thresholds = {
        "Nuclei_Intensity_MassDisplacement_DNA": 1.5,
        "Nuclei_AreaShape_Compactness": 1.0,
    }

oversegmented_nuclei_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

if render_images:
    # MUST SET DATA AS DATAFRAME FOR OUTLINE DIR TO WORK
    oversegmented_nuclei_outliers_cdf = CytoDataFrame(
        data=pd.DataFrame(oversegmented_nuclei_outliers),
        data_outline_context_dir=outline_context_dir,
        segmentation_file_regex=outline_to_orig_mapping,
        display_options={
            "center_dot": False,
            "outline_color": (180, 30, 180),  # magenta
            "brightness": 1,
        },
    )[
        [
            "Nuclei_Intensity_MassDisplacement_DNA",
            "Nuclei_AreaShape_Compactness",
            "Image_FileName_OrigDNA",
        ]
    ]

    print(oversegmented_nuclei_outliers_cdf.shape)
    oversegmented_nuclei_outliers_cdf.sort_values(
        by="Nuclei_AreaShape_Compactness", ascending=False
    ).head(5).T
    # oversegmented_nuclei_outliers_cdf.sample(n=5).T
else:
    print(
        f"{len(oversegmented_nuclei_outliers)} oversegmented nuclei outliers found "
        "(image rendering skipped, render_images=False)"
    )


# In[ ]:


# find non-round nuclei (poorly segmented)
feature_thresholds = {
    "Nuclei_AreaShape_Solidity": -2.2,
}

poorly_segmented_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

if render_images:
    # MUST SET DATA AS DATAFRAME FOR OUTLINE DIR TO WORK
    poorly_segmented_outliers_cdf = CytoDataFrame(
        data=pd.DataFrame(poorly_segmented_outliers),
        data_outline_context_dir=outline_context_dir,
        segmentation_file_regex=outline_to_orig_mapping,
        display_options={
            "center_dot": False,
            "outline_color": (180, 30, 180),  # magenta
            "brightness": 1,
        },
    )[
        [
            "Nuclei_AreaShape_Solidity",
            "Image_FileName_OrigDNA",
        ]
    ]

    print(poorly_segmented_outliers_cdf.shape)
    poorly_segmented_outliers_cdf.sort_values(
        by="Nuclei_AreaShape_Solidity", ascending=True
    ).head(5).T
    # poorly_segmented_outliers_cdf.sample(n=5).T
else:
    print(
        f"{len(poorly_segmented_outliers)} poorly segmented nuclei outliers found "
        "(image rendering skipped, render_images=False)"
    )


# ### Scatterplot of mass displacement to compactness

# In[9]:


# Set the default value to 'inlier'
plate_df["Outlier_Status"] = "Single-cell passed QC"

# Mark outliers from both nuclei clusters and poorly segmented nuclei
combined_idx = pd.Index(oversegmented_nuclei_outliers.index).union(
    pd.Index(poorly_segmented_outliers.index)
)
plate_df.loc[plate_df.index.isin(combined_idx), "Outlier_Status"] = (
    "Single-cell failed QC"
)

# Create scatter plot
plt.figure(figsize=(10, 6))
plot = sns.scatterplot(
    data=plate_df,
    x="Nuclei_Intensity_MassDisplacement_DNA",
    y="Nuclei_AreaShape_Compactness",
    hue="Outlier_Status",
    palette={
        "Single-cell passed QC": "#006400",
        "Single-cell failed QC": "#990090",
    },  # Specify colors
    alpha=0.6,
)

plt.title(f"Nuclei Compactness vs. Nuclei Mass Displacement for {plate}")
plt.xlabel("Nuclei Mass Displacement")
plt.ylabel("Nuclei Compactness")
plt.tight_layout()

# Show the legend
plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0), prop={"size": 10})

# Save figure
plt.savefig(pathlib.Path(f"{qc_fig_dir}/{plate}_nuclei_outliers.png"), dpi=500)

plt.show()


# ## Mis-segmented cells due to high confluence (segmentation for cells around the nuclei)

# In[10]:


# set compartment for segmentation mask
compartment = "Cells"

# channels to include for cytodataframe visualization
channels = ["OrigDNA", "OrigActin"]

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    *[f"Metadata_{compartment}_Location_Center_{axis}" for axis in ("X", "Y")],
    *[f"Image_FileName_{ch}" for ch in channels],
    *[f"Image_PathName_{ch}" for ch in channels],
    *[
        f"{compartment}_AreaShape_BoundingBox{bound}_{axis}"
        for bound in ("Maximum", "Minimum")
        for axis in ("X", "Y")
    ],
]


# In[11]:


# create an outline and orig mapping dictionary to map original images to outlines
# note: we turn off formatting here to avoid the key-value pairing definition
# from being reformatted by black, which is normally preferred.
# fmt: off
outline_to_orig_mapping = {}
for record in plate_df[
    [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]
].to_dict(orient="records"):
    site_raw = str(record["Image_Metadata_Site"])
    # extract digits from the site string (e.g., 'f09' -> '09')
    site_digits = re.sub(r"\D", "", site_raw)
    if site_digits == "":
        site_fmt = site_raw
    else:
        site_fmt = f"{int(site_digits):02d}"

    key = rf"{compartment}Outlines_{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}_{record['Image_Metadata_Site']}.tiff"
    value = rf"{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}f{site_fmt}d\d+.TIF"
    outline_to_orig_mapping[key] = value
# fmt: on

next(iter(outline_to_orig_mapping.items()))


# In[ ]:


# find under-segmented cells (small cells)
feature_thresholds = {
    "Cells_AreaShape_Area": -1.1,
}

small_cells_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

if render_images:
    # MUST SET DATA AS DATAFRAME FOR OUTLINE DIR TO WORK
    small_cells_outliers_cdf = CytoDataFrame(
        data=pd.DataFrame(small_cells_outliers),
        data_outline_context_dir=outline_context_dir,
        segmentation_file_regex=outline_to_orig_mapping,
        display_options={
            "center_dot": False,
            "outline_color": (180, 30, 180),  # magenta
            "brightness": 1,
        },
    )[
        [
            "Cells_AreaShape_Area",
            "Image_FileName_OrigActin",
        ]
    ]

    print(small_cells_outliers_cdf.shape)
    small_cells_outliers_cdf.sort_values(
        by="Cells_AreaShape_Area", ascending=False
    ).head(5).T
    # small_cells_outliers_cdf.sample(n=5).T
else:
    print(
        f"{len(small_cells_outliers)} small cell outliers found "
        "(image rendering skipped, render_images=False)"
    )


# In[13]:


# Set default value
plate_df["Outlier_Status"] = "Single-cell passed QC"

# Mark outliers from under-segmented cells
combined_idx = pd.Index(small_cells_outliers.index)
plate_df.loc[plate_df.index.isin(combined_idx), "Outlier_Status"] = (
    "Single-cell failed QC"
)

# Create plot
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=plate_df,
    x="Cells_AreaShape_Area",
    hue="Outlier_Status",
    palette={
        "Single-cell passed QC": "#006400",
        "Single-cell failed QC": "#990090",
    },
    element="step",
    bins=50,
    alpha=0.5,
    kde=True,  # adds KDE overlay
)

plt.title(f"Distribution of Cell Area for {plate}")
plt.xlabel("Cell Area")
plt.ylabel("Count")

# Customize the Seaborn-generated legend instead of creating a new one
ax.legend_.set_title("Outlier Status")
ax.legend_.set_bbox_to_anchor((1.0, 1.0))
for text in ax.legend_.get_texts():
    text.set_fontsize(10)
ax.legend_.get_title().set_fontsize(11)

plt.tight_layout()
plt.savefig(pathlib.Path(f"{qc_fig_dir}/{plate}_cells_outliers.png"), dpi=500)
plt.show()


# ## Detect blurry cells
# 
# We decided to use texture in the nucleus (nucleus compartment) and actin (cells compartment) to identify out-of-focus cells as it is expected that the pixel intensities will be homogenous across the cell (lack of texture).

# In[ ]:


# find blurry cells
feature_thresholds = {
    "Nuclei_Texture_InfoMeas1_DNA_3_02_256": -1.0,
    "Cells_Texture_InfoMeas1_Actin_3_02_256": -1.0,
}

blurry_cells_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

if render_images:
    # MUST SET DATA AS DATAFRAME FOR OUTLINE DIR TO WORK
    blurry_cells_outliers_cdf = CytoDataFrame(
        data=pd.DataFrame(blurry_cells_outliers),
        data_outline_context_dir=outline_context_dir,
        segmentation_file_regex=outline_to_orig_mapping,
        display_options={
            "center_dot": True,
            "brightness": 5,
        },
    )[
        [
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Nuclei_Texture_InfoMeas1_DNA_3_02_256",
            "Cells_Texture_InfoMeas1_Actin_3_02_256",
            "Image_FileName_OrigActin",
        ]
    ]

    print(blurry_cells_outliers_cdf.shape)
    # blurry_cells_outliers_cdf.sort_values(
    #     by="Cells_Texture_InfoMeas1_Actin_3_02_256", ascending=False
    # ).head(5).T
    blurry_cells_outliers_cdf.sample(n=5).T
else:
    print(
        f"{len(blurry_cells_outliers)} blurry cell outliers found "
        "(image rendering skipped, render_images=False)"
    )


# In[15]:


# Set the default value to 'inlier'
plate_df["Outlier_Status"] = "Single-cell passed QC"

# Mark outliers from under-segmented cells
combined_idx = pd.Index(blurry_cells_outliers.index)
plate_df.loc[plate_df.index.isin(combined_idx), "Outlier_Status"] = (
    "Single-cell failed QC"
)

# Create scatter plot
plt.figure(figsize=(10, 6))
plot = sns.scatterplot(
    data=plate_df,
    x="Nuclei_Texture_InfoMeas1_DNA_3_02_256",
    y="Cells_Texture_InfoMeas1_Actin_3_02_256",
    hue="Outlier_Status",
    palette={
        "Single-cell passed QC": "#006400",
        "Single-cell failed QC": "#990090",
    },  # Specify colors
    alpha=0.6,
)

plt.title(f"Nuclei Texture vs. Cells Texture for {plate}")
plt.xlabel("Nuclei Texture")
plt.ylabel("Cells Texture")
plt.tight_layout()

# Show the legend
plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0), prop={"size": 10})

# Save figure
plt.savefig(pathlib.Path(f"{qc_fig_dir}/{plate}_blurry_outliers.png"), dpi=500)

plt.show()


# ## Remove all outliers and save cleaned data frame

# In[16]:


# Collect unique outlier indices from all known outlier dataframes in the notebook
outlier_names = [
    "oversegmented_nuclei_outliers",
    "poorly_segmented_outliers",
    "small_cells_outliers",
    "blurry_cells_outliers",
]

outlier_frames = []

for name in outlier_names:
    obj = globals().get(name)
    if obj is None:
        continue

    if isinstance(obj, pd.DataFrame):
        outlier_frames.append(obj)
        continue

    if isinstance(obj, pd.Index):
        outlier_frames.append(pd.DataFrame(index=obj))
        continue

    if isinstance(obj, (list, tuple, set)):
        try:
            candidate = pd.DataFrame(obj)
        except (TypeError, ValueError):
            continue
        if not candidate.empty:
            outlier_frames.append(candidate)
        continue

    if isinstance(obj, dict):
        try:
            candidate = pd.DataFrame(obj)
        except (TypeError, ValueError):
            continue
        if not candidate.empty:
            outlier_frames.append(candidate)

if outlier_frames:
    outlier_indices = pd.Index(
        pd.concat(outlier_frames, sort=False).index.unique()
    )
else:
    outlier_indices = pd.Index([])

print(
    f"Found {len(outlier_indices)} unique outlier indices from: "
    + ", ".join(name for name in outlier_names if globals().get(name) is not None)
)

# Remove rows with outlier indices from the plate DataFrame
plate_df_cleaned = plate_df.drop(outlier_indices)

# Save cleaned data for this plate
metadata_plate = plate_df["Image_Metadata_Plate"].iloc[0]
if metadata_plate != plate:
    raise ValueError(
        f"Loaded plate metadata ({metadata_plate}) does not match requested plate ({plate})."
    )

cleaned_path = cleaned_dir / f"{plate}_cleaned.parquet"
plate_df_cleaned.to_parquet(cleaned_path)

# Verify the result
print(plate_df_cleaned.shape)
plate_df_cleaned.head()


# In[17]:


# Compute overall and per-well QC failure rates using outlier_indices
if "plate_df" not in globals():
    raise NameError(
        "plate_df not found in the notebook namespace. Run the cell that loads the plate first."
    )
if "outlier_indices" not in globals():
    raise NameError(
        "outlier_indices not found. Run the cell that collects outlier indices."
    )

df_all = plate_df
total_cells = len(df_all)

# Ensure outlier_indices is an Index and restrict to indices present in plate_df
out_idx = pd.Index(outlier_indices)
out_idx_in_df = df_all.index.intersection(out_idx)

n_failed = len(out_idx_in_df)
pct_failed = n_failed / total_cells * 100 if total_cells else 0.0

print(f"Total cells: {total_cells}")
print(f"Outlier indices provided: {len(out_idx)}")
print(f"Outlier indices present in plate_df: {n_failed} ({pct_failed:.2f}%)")

# Per-well failure percentages using outlier indices
well_counts = df_all.groupby("Image_Metadata_Well").size().rename("total")
well_failed = (
    df_all.loc[df_all.index.isin(out_idx_in_df)]
    .groupby("Image_Metadata_Well")
    .size()
    .rename("failed")
)

well_stats = (
    pd.concat([well_counts, well_failed], axis=1)
    .fillna(0)
    .astype({"total": int, "failed": int})
)
well_stats["failed_pct"] = well_stats["failed"] / well_stats["total"] * 100

# Sort and show top wells
well_stats = well_stats.sort_values("failed_pct", ascending=False)
top_n = globals().get("top_n", 10)
print(f"\nTop {top_n} wells by % failed:")
print(well_stats.head(top_n).to_string())

# Save summary for later use
plate_qc_summary = {
    "total_cells": total_cells,
    "failed_cells": n_failed,
    "failed_pct": pct_failed,
    "outlier_indices_total": len(out_idx),
    "outlier_indices_used": len(out_idx_in_df),
    "well_stats": well_stats,
}


# ## Save per-plate QC summary across conditions
# 
# We save one combined CSV summarizing, per plate, the percentage of single-cells that failed each QC condition and the overall percentage failed. Each run of this notebook overwrites only the row(s) for the current `plate`, so re-running QC for one plate does not affect the summary rows for other plates.

# In[ ]:


total_cells = len(plate_df)

# Map each QC condition to the percentage of single-cells it flagged as failing
condition_outliers = {
    "Oversegmented_Nuclei_Failed_Pct": oversegmented_nuclei_outliers,
    "Poorly_Segmented_Nuclei_Failed_Pct": poorly_segmented_outliers,
    "Small_Cells_Failed_Pct": small_cells_outliers,
    "Blurry_Cells_Failed_Pct": blurry_cells_outliers,
}

summary_row = {"Plate": plate, "Total_Cells": total_cells}
for column_name, outliers in condition_outliers.items():
    n_failed = len(plate_df.index.intersection(pd.Index(outliers.index)))
    summary_row[column_name] = (n_failed / total_cells * 100) if total_cells else 0.0

# Overall failure percentage uses the de-duplicated union of all outlier indices
# (a cell can fail more than one condition, so this is not the sum of the columns above)
n_failed_overall = len(plate_df.index.intersection(pd.Index(outlier_indices)))
summary_row["Overall_Failed_Pct"] = (
    n_failed_overall / total_cells * 100 if total_cells else 0.0
)

new_summary_row_df = pd.DataFrame([summary_row])

# Single combined summary CSV shared across all plates in this dataset
# (HLHS runs save to data/hlhs/qc_summary.csv, matching the converted/cleaned profile dirs)
qc_summary_path = qc_summary_dir / "qc_summary.csv"

if qc_summary_path.exists():
    qc_summary_df = pd.read_csv(qc_summary_path)
    # Drop any existing row for this plate so this run's results overwrite it
    qc_summary_df = qc_summary_df[qc_summary_df["Plate"] != plate]
    qc_summary_df = pd.concat([qc_summary_df, new_summary_row_df], ignore_index=True)
else:
    qc_summary_df = new_summary_row_df

qc_summary_df = qc_summary_df.sort_values("Plate").reset_index(drop=True)
qc_summary_df.to_csv(qc_summary_path, index=False)

print(f"Saved QC summary for {plate} to {qc_summary_path}")
qc_summary_df

