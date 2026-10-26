#!/usr/bin/env python
# coding: utf-8

# # Generate well-level correlation heatmaps
# 
# We use the QC'd well-level (bulk) profiles from `3.preprocessing_profiles/3.bulk_processing` (single cells aggregated per well with the median, then annotated, normalized, and feature selected with pycytominer) and use Pearson's correlation to derive scores between wells.
# All plates in the run are combined using the features shared across plates, and we also generate heatmaps per plate and per heart.
# Set `hlhs_run = True` to use the HLHS plates instead of the default subtyping plates.

# In[1]:


import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from pycytominer.cyto_utils import infer_cp_features


# Optional CLI fallback for direct script-style execution.
def _str_to_bool(value: str) -> bool:
    """
    Convert a string to a boolean value.

    Args:
        value (str): The string to convert.

    Returns:
        bool: True if the string represents a true value, False otherwise.
    """
    return str(value).strip().lower() in ("1", "true", "yes")


def plot_well_heatmap(
    well_df: pd.DataFrame,
    annotation_colors: dict,
    title: str,
    save_path: pathlib.Path,
    show_well_labels: bool = False,
    show_legends: bool = True,
    cluster: bool = False,
) -> pd.DataFrame:
    """
    Plot a heatmap of the Pearson correlation between wells. By default there is no clustering
    and the wells are kept in the same order as in well_df.

    Args:
        well_df (pd.DataFrame): Well-level profiles with a Metadata_Well_ID column.
        annotation_colors (dict): Maps an annotation name to a tuple of
            (metadata column, {category: color}) used for the row/column colors and legends.
        title (str): Figure title.
        save_path (pathlib.Path): Path to save the figure.
        show_well_labels (bool): Whether to label the rows/columns with the well IDs.
        show_legends (bool): Whether to add a legend for each annotation.
        cluster (bool): Whether to order the wells with hierarchical clustering (with dendrograms).

    Returns:
        pd.DataFrame: The well-by-well correlation matrix.
    """
    features = infer_cp_features(well_df)
    well_corr_matrix = well_df.set_index("Metadata_Well_ID")[features].T.corr()

    # Color bars for each metadata annotation, aligned to the correlation matrix
    meta_df = well_df.set_index("Metadata_Well_ID")
    row_colors = pd.DataFrame(
        {
            name: meta_df[column].map(colors)
            for name, (column, colors) in annotation_colors.items()
        }
    ).reindex(well_corr_matrix.index)

    g = sns.clustermap(
        well_corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=(10, 10),
        linewidths=0,
        xticklabels=show_well_labels,
        yticklabels=show_well_labels,
        row_cluster=cluster,
        col_cluster=cluster,
        dendrogram_ratio=(0.08, 0.08) if cluster else (0.01, 0.01),
        cbar_pos=(1.02, 0.08, 0.02, 0.3),
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    # The color bars are labeled on the top only, so remove the bottom left labels
    g.ax_row_colors.set_xticks([])
    if show_well_labels:
        # Plate is shown by the color bar, so only label with the well
        well_order = (
            well_corr_matrix.index[g.dendrogram_row.reordered_ind]
            if cluster
            else well_corr_matrix.index
        )
        well_labels = meta_df.loc[well_order, "Metadata_Well"]
        g.ax_heatmap.set_xticklabels(well_labels, fontsize=9)
        g.ax_heatmap.set_yticklabels(well_labels, fontsize=9, rotation=0)
    # Center the title over the heatmap
    heatmap_pos = g.ax_heatmap.get_position()
    g.figure.suptitle(
        title,
        x=(heatmap_pos.x0 + heatmap_pos.x1) / 2,
        y=0.99,
        va="bottom",
        fontsize=16,
    )

    # Find the right edge of the heatmap (including tick and color bar labels)
    # so the legends and colorbar sit right next to it
    fig = g.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    heatmap_bbox = g.ax_heatmap.get_tightbbox(renderer).transformed(to_fig)
    col_colors_bbox = g.ax_col_colors.get_tightbbox(renderer).transformed(to_fig)
    legend_x = max(heatmap_bbox.x1, col_colors_bbox.x1) + 0.02
    legend_top = g.ax_col_colors.get_position().y1

    # Stack one legend per annotation, measuring each to place the next one below it
    for name, (column, colors) in (annotation_colors.items() if show_legends else []):
        present = [c for c in colors if c in set(meta_df[column])]
        leg = fig.legend(
            handles=[mpatches.Patch(color=colors[c], label=c) for c in present],
            title=name.replace("\n", " "),
            loc="upper left",
            bbox_to_anchor=(legend_x, legend_top),
            frameon=False,
            ncol=2 if len(present) > 6 else 1,
            fontsize=9,
            title_fontsize=10,
            alignment="left",
            borderaxespad=0,
            handlelength=1.2,
            columnspacing=1.0,
        )
        leg.get_title().set_fontweight("bold")
        fig.canvas.draw()
        legend_top = leg.get_window_extent(renderer).transformed(to_fig).y0 - 0.03

    # Place the correlation colorbar below the legends
    cbar_height = min(0.25, legend_top - g.ax_heatmap.get_position().y0)
    g.ax_cbar.set_position([legend_x, legend_top - cbar_height, 0.02, cbar_height])
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label("Pearson correlation", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    g.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return well_corr_matrix


def summarize_replicate_correlation(
    well_corr_matrix: pd.DataFrame, well_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarize the correlation between wells from the same heart versus different hearts.

    Args:
        well_corr_matrix (pd.DataFrame): Well-by-well correlation matrix.
        well_df (pd.DataFrame): Well-level profiles with a Metadata_Well_ID column.

    Returns:
        pd.DataFrame: Mean, median, and number of well pairs per comparison type.
    """
    meta_df = well_df.set_index("Metadata_Well_ID")
    pairs = (
        well_corr_matrix.rename_axis(index="well_a", columns="well_b")
        .stack()
        .rename("pearson_correlation")
        .reset_index()
    )
    # Keep each pair of wells once
    pairs = pairs[pairs["well_a"] < pairs["well_b"]]

    for side in ["a", "b"]:
        pairs[f"heart_{side}"] = pairs[f"well_{side}"].map(
            meta_df["Metadata_heart_number"]
        )
        pairs[f"plate_{side}"] = pairs[f"well_{side}"].map(meta_df["Metadata_Plate"])

    same_heart = pairs["heart_a"] == pairs["heart_b"]
    same_plate = pairs["plate_a"] == pairs["plate_b"]
    pairs["comparison"] = (
        same_heart.map({True: "Same heart", False: "Different heart"})
        + ", "
        + same_plate.map({True: "same plate", False: "different plate"})
    )

    return (
        pairs.groupby("comparison")["pearson_correlation"]
        .agg(["mean", "median", "count"])
        .rename(columns={"count": "n_well_pairs"})
        .sort_values("mean", ascending=False)
    )


# In[2]:


# Set if this run is for the HLHS plates (will be updated by papermill)
hlhs_run = False


# In[3]:


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--hlhs-run", dest="hlhs_run", type=_str_to_bool, default=hlhs_run
)
args, _ = parser.parse_known_args(sys.argv[1:])
hlhs_run = args.hlhs_run

# Directory with the bulk (well-level) profiles and figure directory
# (HLHS plates are in a subfolder of the default subtyping plates)
if hlhs_run:
    data_dir = pathlib.Path("../3.preprocessing_profiles/data/bulk_profiles/hlhs")
    output_dir = pathlib.Path("figures/heatmaps/hlhs_plates")
else:
    data_dir = pathlib.Path("../3.preprocessing_profiles/data/bulk_profiles/")
    output_dir = pathlib.Path("figures/heatmaps/subtyping_plates")
output_dir.mkdir(parents=True, exist_ok=True)

# Figure directories for the heatmaps generated per plate and per heart
per_plate_output_dir = output_dir / "per_plate"
per_plate_output_dir.mkdir(parents=True, exist_ok=True)
per_heart_output_dir = output_dir / "per_heart"
per_heart_output_dir.mkdir(parents=True, exist_ok=True)
per_heart_per_plate_output_dir = per_heart_output_dir / "split_by_plate"
per_heart_per_plate_output_dir.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette for the heart failure types
failure_type_colors = {
    "Healthy": "#009E73",
    "DCM": "#0072B2",
    "ICM": "#E69F00",
}
if hlhs_run:
    failure_type_colors["HLHS"] = "#CC79A7"


# In[4]:


# Load in the feature selected bulk profiles for all plates in the run
fs_paths = sorted(data_dir.glob("*_bulk_feature_selected.parquet"))
if not fs_paths:
    raise FileNotFoundError(f"No feature selected parquet files found in {data_dir}")

plate_dfs = [pd.read_parquet(path) for path in fs_paths]

# Feature selection is performed per plate, so only keep the columns shared across all plates
well_df = pd.concat(plate_dfs, join="inner", ignore_index=True)

# Healthy hearts have no heart failure type, so label them as Healthy
well_df["Metadata_heart_failure_type"] = well_df["Metadata_heart_failure_type"].fillna(
    "Healthy"
)

pd.DataFrame(
    {
        "file": [path.name for path in fs_paths],
        "n_wells": [df.shape[0] for df in plate_dfs],
        "n_cells": [df["Metadata_Object_Count"].sum() for df in plate_dfs],
        "n_features": [len(infer_cp_features(df)) for df in plate_dfs],
    }
)


# In[ ]:


# Well IDs repeat across plates, so make a unique ID per plate and well
well_df["Metadata_heart_number"] = well_df["Metadata_heart_number"].astype(str)
well_df["Metadata_Well_ID"] = well_df["Metadata_Plate"] + "_" + well_df["Metadata_Well"]

# Number of wells per heart and plate
well_df.pivot_table(
    index=["Metadata_heart_failure_type", "Metadata_heart_number"],
    columns="Metadata_Plate",
    values="Metadata_Well",
    aggfunc="count",
)


# In[6]:


# Order the hearts by heart failure type and then heart number
heart_order = (
    well_df[["Metadata_heart_failure_type", "Metadata_heart_number"]]
    .drop_duplicates()
    .assign(
        failure_type_sort=lambda df: df["Metadata_heart_failure_type"].map(
            list(failure_type_colors).index
        ),
        heart_sort=lambda df: pd.to_numeric(
            df["Metadata_heart_number"], errors="coerce"
        ),
    )
    .sort_values(["failure_type_sort", "heart_sort", "Metadata_heart_number"])[
        "Metadata_heart_number"
    ]
    .tolist()
)

# Color palettes for the plates and hearts (same color for a plate/heart in every figure)
plates = sorted(well_df["Metadata_Plate"].unique())
plate_colors = dict(zip(plates, ["#4a3aa7", "#e87ba4", "#008300"]))
heart_colors = dict(
    zip(heart_order, sns.color_palette("Paired", n_colors=len(heart_order)))
)

# Order the wells by heart failure type, heart, plate, and then well
# (the heatmaps are not clustered, so the wells are plotted in this order)
well_df = well_df.sort_values(
    ["Metadata_heart_number", "Metadata_Plate", "Metadata_Well"],
    key=lambda col: col.map(heart_order.index)
    if col.name == "Metadata_heart_number"
    else col,
).reset_index(drop=True)


# ## Correlation between all wells across plates

# In[7]:


all_plates_corr_matrix = plot_well_heatmap(
    well_df,
    annotation_colors={
        "Plate": ("Metadata_Plate", plate_colors),
        "Heart failure\ntype": ("Metadata_heart_failure_type", failure_type_colors),
        "Heart number": ("Metadata_heart_number", heart_colors),
    },
    title="Pearson correlation between wells (all plates)",
    save_path=output_dir / "well_correlation_heatmap_all_plates.png",
)

summarize_replicate_correlation(all_plates_corr_matrix, well_df)


# ## Correlation between wells per plate
# 
# For each plate, we generate one heatmap with the wells ordered by heart failure type and heart, and an extra heatmap with the wells ordered by hierarchical clustering.

# In[8]:


for plate, plate_well_df in well_df.groupby("Metadata_Plate"):
    plate_corr_matrix = plot_well_heatmap(
        plate_well_df,
        annotation_colors={
            "Heart failure\ntype": ("Metadata_heart_failure_type", failure_type_colors),
            "Heart number": ("Metadata_heart_number", heart_colors),
        },
        title=f"Pearson correlation between wells ({plate})",
        save_path=per_plate_output_dir / f"well_correlation_heatmap_{plate}.png",
    )
    display(summarize_replicate_correlation(plate_corr_matrix, plate_well_df))

    # Extra heatmap with hierarchical clustering
    plot_well_heatmap(
        plate_well_df,
        annotation_colors={
            "Heart failure\ntype": ("Metadata_heart_failure_type", failure_type_colors),
            "Heart number": ("Metadata_heart_number", heart_colors),
        },
        title=f"Pearson correlation between wells ({plate}, hierarchical clustering)",
        save_path=per_plate_output_dir / f"well_correlation_heatmap_{plate}_clustered.png",
        cluster=True,
    )


# ## Correlation between wells per heart
# 
# Each heart has replicate wells across plates, so the wells are colored by plate to check if the wells group by plate.

# In[9]:


for heart_num in heart_order:
    heart_well_df = well_df[well_df["Metadata_heart_number"] == heart_num]
    failure_type = heart_well_df["Metadata_heart_failure_type"].iloc[0]
    plot_well_heatmap(
        heart_well_df,
        annotation_colors={"Plate": ("Metadata_Plate", plate_colors)},
        title=f"Pearson correlation between wells (Heart #{heart_num}, {failure_type})",
        save_path=per_heart_output_dir
        / f"well_correlation_heatmap_heart_{heart_num}.png",
        show_well_labels=True,
    )


# ## Correlation between wells per heart, split by plate

# In[10]:


for heart_num in heart_order:
    heart_well_df = well_df[well_df["Metadata_heart_number"] == heart_num]
    failure_type = heart_well_df["Metadata_heart_failure_type"].iloc[0]
    for plate, heart_plate_well_df in heart_well_df.groupby("Metadata_Plate"):
        plate_output_dir = per_heart_per_plate_output_dir / plate
        plate_output_dir.mkdir(parents=True, exist_ok=True)
        plot_well_heatmap(
            heart_plate_well_df,
            annotation_colors={"Plate": ("Metadata_Plate", plate_colors)},
            title=f"Pearson correlation between wells (Heart #{heart_num}, {failure_type})\n{plate}",
            save_path=plate_output_dir / f"well_correlation_heatmap_heart_{heart_num}.png",
            show_well_labels=True,
            # All wells are from one plate (shown in the title), so no plate legend needed
            show_legends=False,
        )

