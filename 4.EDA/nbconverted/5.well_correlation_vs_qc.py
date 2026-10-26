#!/usr/bin/env python
# coding: utf-8

# # Well replicate correlation versus single-cell QC failure rate
# 
# To find which hearts have the most consistent wells and the fewest segmentation failures, we compare two per-well metrics:
# 
# 1. **Replicate correlation**: how similar a well is to the other wells from the same heart. Each heart has several wells on a plate, and each well has one bulk (well-level) profile from `3.preprocessing_profiles/3.bulk_processing`. For each well, we compute the Pearson correlation between its profile and the profile of each other well from the same heart on the same plate, then average those correlations. For example, if heart 7 has wells B02, C05, D08, E03, and F10 on a plate, the replicate correlation for B02 is the mean of its correlations with C05, D08, E03, and F10. A value near 1 means the wells look the same, and a value near 0 means they are no more similar than wells from different hearts.
# 2. **QC failure rate**: the percent of single cells in the well removed by the single-cell QC (cells before QC from the converted profiles and after QC from the cleaned profiles).
# 
# Hearts in the top left of the plot (high replicate correlation, low QC failure rate) have the best quality wells.
# Set `hlhs_run = True` to use the HLHS plates instead of the default subtyping plates.

# In[ ]:


import argparse
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from IPython.display import display
from matplotlib.lines import Line2D
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


def get_replicate_correlation(bulk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean Pearson correlation between each well and the other wells from the same heart
    (all wells must be from the same plate).

    Args:
        bulk_df (pd.DataFrame): Feature selected bulk profiles for one plate.

    Returns:
        pd.DataFrame: One row per well with the replicate correlation and number of replicate wells.
    """
    features = infer_cp_features(bulk_df)
    corr_matrix = bulk_df.set_index("Metadata_Well")[features].T.corr()
    hearts = bulk_df.set_index("Metadata_Well")["Metadata_heart_number"]

    results = []
    for well, heart in hearts.items():
        replicate_wells = hearts.index[(hearts == heart) & (hearts.index != well)]
        results.append(
            {
                "Metadata_Well": well,
                "replicate_correlation": corr_matrix.loc[well, replicate_wells].mean(),
                "n_replicate_wells": len(replicate_wells),
            }
        )
    return pd.DataFrame(results)


def count_cells_per_well(profile_path: pathlib.Path) -> pd.Series:
    """
    Count the single cells per well in a profile (only the well column is loaded).

    Args:
        profile_path (pathlib.Path): Path to a converted or cleaned single-cell profile.

    Returns:
        pd.Series: Number of single cells per well.
    """
    return (
        pq.read_table(profile_path, columns=["Image_Metadata_Well"])
        .to_pandas()["Image_Metadata_Well"]
        .value_counts()
    )

def label_points_without_overlap(
    ax: plt.Axes, x: pd.Series, y: pd.Series, labels: pd.Series, min_gap: float = 0.045
) -> None:
    """
    Label points to the right of each point, moving labels up or down when they would overlap another
    label (a thin line connects a moved label to its point).

    Args:
        ax (plt.Axes): Axes with the points (the axis limits must already be set).
        x (pd.Series): X values of the points.
        y (pd.Series): Y values of the points.
        labels (pd.Series): Label per point.
        min_gap (float): Minimum distance between labels as a fraction of the axes.
    """
    to_axes = ax.transData + ax.transAxes.inverted()
    points = to_axes.transform(np.column_stack([x, y]))

    placed = []
    for idx in np.argsort(points[:, 1]):
        point_x, point_y = points[idx]
        label_x, label_y = point_x + 0.015, point_y
        # Try positions closest to the point first (alternating up and down)
        for step in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            candidate_y = point_y + step * min_gap
            if all(
                abs(candidate_y - other_y) >= min_gap or abs(label_x - other_x) >= 0.08
                for other_x, other_y in placed
            ):
                label_y = candidate_y
                break
        placed.append((label_x, label_y))
        ax.annotate(
            labels.iloc[idx],
            xy=(point_x, point_y),
            xytext=(label_x, label_y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333",
            # White halo so the labels are readable on top of the SD lines
            path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
            arrowprops=(
                dict(arrowstyle="-", color="#999999", linewidth=0.7, shrinkA=0, shrinkB=5)
                if label_y != point_y
                else None
            ),
        )

def summarize_hearts(well_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the replicate correlation and QC failure rate per heart across its wells.

    Args:
        well_df (pd.DataFrame): Replicate correlation and QC failure rate per well.

    Returns:
        pd.DataFrame: Mean and SD of both metrics per heart, sorted by the mean replicate correlation.
    """
    return (
        well_df.groupby(["Metadata_heart_failure_type", "Metadata_heart_number"])
        .agg(
            mean_replicate_correlation=("replicate_correlation", "mean"),
            sd_replicate_correlation=("replicate_correlation", "std"),
            mean_qc_failed_pct=("qc_failed_pct", "mean"),
            sd_qc_failed_pct=("qc_failed_pct", "std"),
            n_wells=("Metadata_Well", "size"),
            n_cells_after_qc=("n_cells_after_qc", "sum"),
        )
        .reset_index()
        .sort_values("mean_replicate_correlation", ascending=False)
        .reset_index(drop=True)
    )


def plot_well_quality(
    well_df: pd.DataFrame,
    heart_df: pd.DataFrame,
    heart_colors: dict,
    failure_type_markers: dict,
    title: str,
    save_path: pathlib.Path,
    female_hearts: list = [],
) -> None:
    """
    Plot the replicate correlation versus the QC failure rate per well (left) and per heart (right),
    colored by heart number and shaped by heart failure type.

    Args:
        well_df (pd.DataFrame): Replicate correlation and QC failure rate per well.
        heart_df (pd.DataFrame): Mean and SD of both metrics per heart (from summarize_hearts).
        heart_colors (dict): Color per heart number (in the order to show in the legend).
        failure_type_markers (dict): Marker per heart failure type.
        title (str): Name of the plates in the figure title.
        save_path (pathlib.Path): Path to save the figure.
        female_hearts (list): Heart numbers from female patients, marked with * in the labels.
    """
    # Mark the female hearts with * in the labels and legend
    heart_labels = {
        heart_number: f"{heart_number}*" if heart_number in female_hearts else heart_number
        for heart_number in heart_colors
    }

    # The heart panel is not shared with the well panel, so it zooms in on the heart means
    fig, (well_ax, heart_ax) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left: every well, colored by heart number and shaped by heart failure type
    for (heart_number, failure_type), group_df in well_df.groupby(
        ["Metadata_heart_number", "Metadata_heart_failure_type"]
    ):
        well_ax.scatter(
            group_df["qc_failed_pct"],
            group_df["replicate_correlation"],
            color=heart_colors[heart_number],
            marker=failure_type_markers[failure_type],
            s=50,
            alpha=0.85,
            # Thin dark outline so the pale colors (e.g., heart 44) are easy to see
            edgecolor="#333333",
            linewidth=0.6,
        )
    well_ax.set_title("Per well", fontsize=14, loc="left")

    # Right: mean and SD per heart, labeled with the heart number
    for _, heart in heart_df.iterrows():
        error_bars = heart_ax.errorbar(
            heart["mean_qc_failed_pct"],
            heart["mean_replicate_correlation"],
            xerr=heart["sd_qc_failed_pct"],
            yerr=heart["sd_replicate_correlation"],
            fmt=failure_type_markers[heart["Metadata_heart_failure_type"]],
            color=heart_colors[heart["Metadata_heart_number"]],
            markersize=11,
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            elinewidth=1.5,
            alpha=0.9,
            capsize=0,
        )
        # Thin dark outline on the SD lines so the pale colors (e.g., heart 44) are easy to see
        for sd_lines in error_bars.lines[2]:
            sd_lines.set_path_effects(
                [path_effects.Stroke(linewidth=2.7, foreground="#333333"), path_effects.Normal()]
            )
    heart_ax.set_title("Per heart (mean \u00b1 SD across wells)", fontsize=14, loc="left")

    # Set the axis limits to fit the full SD lines of every heart
    for axis, metric in [("x", "qc_failed_pct"), ("y", "replicate_correlation")]:
        sd = heart_df[f"sd_{metric}"].fillna(0)
        low = (heart_df[f"mean_{metric}"] - sd).min()
        high = (heart_df[f"mean_{metric}"] + sd).max()
        padding = 0.05 * (high - low)
        getattr(heart_ax, f"set_{axis}lim")(low - padding, high + padding)
    label_points_without_overlap(
        heart_ax,
        heart_df["mean_qc_failed_pct"],
        heart_df["mean_replicate_correlation"],
        heart_df["Metadata_heart_number"].map(heart_labels),
    )

    for ax in (well_ax, heart_ax):
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Single cells removed by QC (%)", fontsize=12)
        ax.grid(color="#e9e8e4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    well_ax.set_ylabel(
        "Replicate correlation (Pearson r)",
        fontsize=12,
    )

    # Legends for the heart number colors (shaped by the heart's failure type) and failure type shapes
    hearts_in_plot = set(well_df["Metadata_heart_number"])
    heart_failure_types = well_df.groupby("Metadata_heart_number")[
        "Metadata_heart_failure_type"
    ].first()
    heart_handles = [
        Line2D(
            [],
            [],
            marker=failure_type_markers[heart_failure_types[heart_number]],
            linestyle="",
            markersize=9,
            color=color,
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            label=heart_labels[heart_number],
        )
        for heart_number, color in heart_colors.items()
        if heart_number in hearts_in_plot
    ]
    failure_type_handles = [
        Line2D([], [], marker=marker, linestyle="", markersize=9, color="#777777", label=failure_type)
        for failure_type, marker in failure_type_markers.items()
        if failure_type in set(heart_failure_types)
    ]
    fig.legend(
        handles=failure_type_handles,
        title="Heart failure type",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.88),
        frameon=False,
        alignment="left",
    )
    fig.legend(
        handles=heart_handles,
        title="Heart number",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.6),
        frameon=False,
        alignment="left",
        ncol=2,
    )
    # Note for the * when there are female hearts in the plot
    if hearts_in_plot & set(female_hearts):
        fig.text(1.0, 0.12, "* female heart", fontsize=10, color="#555555")

    fig.suptitle(
        f"Well replicate correlation versus single-cell QC failure rate ({title})\n"
        "Best quality = top left (consistent wells and few cells removed by QC)",
        fontsize=15,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


# In[ ]:


# Set if this run is for the HLHS plates (will be updated by papermill)
hlhs_run = False


# In[ ]:


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--hlhs-run", dest="hlhs_run", type=_str_to_bool, default=hlhs_run
)
args, _ = parser.parse_known_args(sys.argv[1:])
hlhs_run = args.hlhs_run

# Directories with the profiles (HLHS plates are in a subfolder of the default subtyping plates)
profiles_dir = pathlib.Path("../3.preprocessing_profiles/data").resolve(strict=True)
run_subfolder = "hlhs" if hlhs_run else ""
bulk_dir = profiles_dir / "bulk_profiles" / run_subfolder
converted_dir = profiles_dir / "converted_profiles" / run_subfolder
cleaned_dir = profiles_dir / "cleaned_profiles" / run_subfolder

# Output directories for the figure and results
run_name = "hlhs_plates" if hlhs_run else "subtyping_plates"
figure_dir = pathlib.Path("figures/well_quality")
figure_dir.mkdir(parents=True, exist_ok=True)
results_dir = pathlib.Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

# Shape per heart failure type (also the order of the heart failure types for sorting the hearts)
failure_type_markers = {"Healthy": "o", "DCM": "s", "ICM": "D"}
if hlhs_run:
    failure_type_markers["HLHS"] = "^"

# Heart numbers from female patients (only on the subtyping plates), marked with * in the plots
female_hearts = [] if hlhs_run else ["32", "64"]


# ## Compute the replicate correlation and QC failure rate per well

# In[ ]:


bulk_paths = sorted(bulk_dir.glob("*_bulk_feature_selected.parquet"))
if not bulk_paths:
    raise FileNotFoundError(f"No feature selected bulk profiles found in {bulk_dir}")

well_quality_dfs = []
for bulk_path in bulk_paths:
    plate = bulk_path.name.replace("_bulk_feature_selected.parquet", "")
    bulk_df = pd.read_parquet(bulk_path)
    bulk_df["Metadata_heart_number"] = bulk_df["Metadata_heart_number"].astype(str)

    # Number of single cells per well before and after QC
    cell_counts_df = pd.DataFrame(
        {
            "n_cells_before_qc": count_cells_per_well(
                converted_dir / f"{plate}_converted.parquet"
            ),
            "n_cells_after_qc": count_cells_per_well(
                cleaned_dir / f"{plate}_cleaned.parquet"
            ),
        }
    ).fillna(0)
    cell_counts_df["qc_failed_pct"] = 100 * (
        1 - cell_counts_df["n_cells_after_qc"] / cell_counts_df["n_cells_before_qc"]
    )

    well_quality_dfs.append(
        bulk_df[
            [
                "Metadata_Plate",
                "Metadata_Well",
                "Metadata_heart_number",
                "Metadata_heart_failure_type",
            ]
        ]
        .merge(get_replicate_correlation(bulk_df), on="Metadata_Well")
        .merge(cell_counts_df, left_on="Metadata_Well", right_index=True, how="left")
    )

well_quality_df = pd.concat(well_quality_dfs, ignore_index=True)

# Healthy hearts have no heart failure type, so label them as Healthy
well_quality_df["Metadata_heart_failure_type"] = well_quality_df[
    "Metadata_heart_failure_type"
].fillna("Healthy")

# Every well should have cells before QC
assert well_quality_df["n_cells_before_qc"].gt(0).all()

well_quality_df.to_csv(results_dir / f"well_quality_{run_name}.csv", index=False)

print(well_quality_df.shape)
well_quality_df.head()


# ## Summarize per heart
# 
# Hearts are sorted by the mean replicate correlation across all of their wells (all plates).

# In[ ]:


heart_quality_df = summarize_hearts(well_quality_df)
heart_quality_df.to_csv(results_dir / f"heart_quality_{run_name}.csv", index=False)

heart_quality_df.round(3)


# ## Plot the replicate correlation versus the QC failure rate (all plates)
# 
# The left panel shows every well (color = heart, shape = heart failure type) and the right panel shows the mean (± SD) per heart across all wells, labeled with the heart number.

# In[ ]:


# Order the hearts by heart failure type and then heart number
heart_order = (
    well_quality_df[["Metadata_heart_failure_type", "Metadata_heart_number"]]
    .drop_duplicates()
    .assign(
        failure_type_sort=lambda df: df["Metadata_heart_failure_type"].map(
            list(failure_type_markers).index
        ),
        heart_sort=lambda df: pd.to_numeric(df["Metadata_heart_number"], errors="coerce"),
    )
    .sort_values(["failure_type_sort", "heart_sort", "Metadata_heart_number"])[
        "Metadata_heart_number"
    ]
    .tolist()
)

# Color per heart (same palette and order as the correlation heatmaps)
heart_colors = dict(zip(heart_order, sns.color_palette("Paired", n_colors=len(heart_order))))

plot_well_quality(
    well_df=well_quality_df,
    heart_df=heart_quality_df,
    heart_colors=heart_colors,
    failure_type_markers=failure_type_markers,
    title=run_name.replace("_", " "),
    save_path=figure_dir / f"well_correlation_vs_qc_{run_name}.png",
    female_hearts=female_hearts,
)


# ## Summarize and plot per plate
# 
# The same metrics per plate, so the mean (± SD) per heart is across the wells from that plate only.

# In[ ]:


# Summarize per heart for each plate
heart_quality_per_plate_df = pd.concat(
    [
        summarize_hearts(plate_well_df).assign(Metadata_Plate=plate)
        for plate, plate_well_df in well_quality_df.groupby("Metadata_Plate")
    ],
    ignore_index=True,
)
heart_quality_per_plate_df.to_csv(
    results_dir / f"heart_quality_per_plate_{run_name}.csv", index=False
)

# Mean replicate correlation and QC failure rate per heart (rows) and plate (columns)
heart_quality_per_plate_df.pivot_table(
    index=["Metadata_heart_failure_type", "Metadata_heart_number"],
    columns="Metadata_Plate",
    values=["mean_replicate_correlation", "mean_qc_failed_pct"],
).round(3)


# In[ ]:


per_plate_figure_dir = figure_dir / f"per_plate_{run_name}"
per_plate_figure_dir.mkdir(parents=True, exist_ok=True)

for plate, plate_well_df in well_quality_df.groupby("Metadata_Plate"):
    plot_well_quality(
        well_df=plate_well_df,
        heart_df=heart_quality_per_plate_df[
            heart_quality_per_plate_df["Metadata_Plate"] == plate
        ],
        heart_colors=heart_colors,
        failure_type_markers=failure_type_markers,
        title=plate,
        save_path=per_plate_figure_dir / f"well_correlation_vs_qc_{plate}.png",
        female_hearts=female_hearts,
    )


# ## Check if the replicate correlation is related to the QC failure rate
# 
# Spearman correlation between the two metrics across the wells and across the hearts, for all plates and per plate.

# In[ ]:


# Groups of wells and hearts to compare (all plates and each plate)
comparison_groups = [("all plates", well_quality_df, heart_quality_df)] + [
    (
        plate,
        plate_well_df,
        heart_quality_per_plate_df[heart_quality_per_plate_df["Metadata_Plate"] == plate],
    )
    for plate, plate_well_df in well_quality_df.groupby("Metadata_Plate")
]

pd.DataFrame(
    [
        {
            "plates": name,
            "well_spearman_rho": well_df["replicate_correlation"].corr(
                well_df["qc_failed_pct"], method="spearman"
            ),
            "n_wells": len(well_df),
            "heart_spearman_rho": heart_df["mean_replicate_correlation"].corr(
                heart_df["mean_qc_failed_pct"], method="spearman"
            ),
            "n_hearts": len(heart_df),
        }
        for name, well_df, heart_df in comparison_groups
    ]
).round(3)

