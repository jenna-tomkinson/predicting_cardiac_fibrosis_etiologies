#!/usr/bin/env python
# coding: utf-8

# # Check the cell counts across hearts

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


# Load in QC profiles from all annotated parquet files (both plates)
data_dir = pathlib.Path("../3.preprocessing_profiles/data/single_cell_profiles")
annotated_files = sorted(data_dir.glob("*_annotated.parquet"))

if not annotated_files:
    raise FileNotFoundError(f"No annotated parquet files found in {data_dir}")

dfs = []
for p in annotated_files:
    df_tmp = pd.read_parquet(p)
    # keep track of source plate/file
    df_tmp["source_plate_file"] = p.name
    dfs.append(df_tmp)

# Concatenate into one profile_df for downstream cells
profile_df = pd.concat(dfs, ignore_index=True)

print("Loaded annotated files:", [p.name for p in annotated_files])
print("Combined Profile DataFrame shape:", profile_df.shape)
profile_df.head()


# In[5]:


# Label hearts without a heart failure subtype (None) as Non-failing
profile_df["Metadata_heart_failure_type"] = profile_df[
    "Metadata_heart_failure_type"
].fillna("Non-failing")

# Compute counts per heart failure subtype, heart, and plate
counts_df = (
    profile_df.groupby(
        ["Metadata_heart_failure_type", "Metadata_heart_number", "Metadata_Plate"]
    )
    .size()
    .reset_index(name="cell_count")
)

# Order the panels with Non-failing first and then the remaining subtypes
subtypes = ["Non-failing"] + sorted(
    set(counts_df["Metadata_heart_failure_type"]) - {"Non-failing"}
)
hearts_per_subtype = {
    subtype: sorted(
        counts_df.loc[
            counts_df["Metadata_heart_failure_type"] == subtype,
            "Metadata_heart_number",
        ].unique(),
        key=lambda heart: (not str(heart).isdigit(), int(heart) if str(heart).isdigit() else str(heart)),
    )
    for subtype in subtypes
}

# One color per plate (colorblind-friendly, same color for a plate in every panel)
plates = sorted(counts_df["Metadata_Plate"].unique())
plate_colors = dict(zip(plates, ["#4B2E83", "#9B1D44", "#C27803"]))  # deep violet, berry crimson, burnt amber
# Text color for the cell count labels inside the bars of each plate
plate_label_colors = dict(zip(plates, ["white", "#1a1a1a", "white"]))

# Female hearts (marked with a star next to the heart number)
female_hearts = [32, 64]

# One panel per subtype, with panel width scaled to the number of hearts
fig, axes = plt.subplots(
    1,
    len(subtypes),
    figsize=(14, 6.5),
    sharey=True,
    gridspec_kw={
        "width_ratios": [len(hearts_per_subtype[subtype]) for subtype in subtypes],
        "wspace": 0.06,
    },
)

for ax, subtype in zip(axes, subtypes):
    hearts = hearts_per_subtype[subtype]
    sns.barplot(
        data=counts_df[counts_df["Metadata_heart_failure_type"] == subtype],
        x="Metadata_heart_number",
        y="cell_count",
        hue="Metadata_Plate",
        order=hearts,
        hue_order=plates,
        palette=plate_colors,
        saturation=1,
        width=0.78,
        edgecolor="white",
        linewidth=1.5,
        legend=False,
        ax=ax,
    )
    # Add the cell count vertically inside the top of each bar
    for container, plate in zip(ax.containers, plates):
        for bar in container:
            ax.annotate(
                f"{bar.get_height():.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, -5),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="top",
                fontsize=10,
                color=plate_label_colors[plate],
            )
    # Add a star next to the heart number of the female hearts
    ax.set_xticks(range(len(hearts)))
    ax.set_xticklabels(
        [f"{heart} ★" if heart in female_hearts else str(heart) for heart in hearts]
    )
    ax.set_title(subtype, fontsize=15, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", length=0, labelsize=13)
    ax.tick_params(axis="y", length=0, labelsize=12, colors="#555555")
    ax.grid(axis="y", color="#e6e6e3", linewidth=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")

axes[0].set_ylabel("Cell count", fontsize=14)
fig.supxlabel("Heart number (★ = female heart)", fontsize=14, y=0.115)
fig.suptitle(
    "Cell counts per heart, split by heart failure subtype",
    fontsize=18,
    x=0.07,
    y=0.93,
    ha="left",
)
fig.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=plate_colors[plate]) for plate in plates],
    labels=plates,
    title="Plate",
    title_fontsize=12,
    fontsize=11,
    ncol=len(plates),
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    frameon=False,
    handlelength=1.2,
    handleheight=1.2,
    columnspacing=1.5,
)
fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.2)

# Save plot
output_plot_dir = pathlib.Path("../4.EDA/figures")
output_plot_dir.mkdir(parents=True, exist_ok=True)
output_plot_path = output_plot_dir / "cell_counts_per_heart_treatment.png"
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
print("Plot saved to:", output_plot_path)
plt.show()


# In[4]:


# Define the column to plot
col = "Cells_Neighbors_NumberOfNeighbors_Adjacent"

# Determine order automatically from the data
order = sorted(profile_df["Metadata_heart_number"].dropna().unique())

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=profile_df.dropna(subset=[col]),
    x="Metadata_heart_number",
    y=col,
    hue="Metadata_Plate",
    order=order,
    showcaps=True,
    fliersize=2,
    boxprops={"alpha": 0.8},
)

plt.xlabel("Heart Treatment")
plt.ylabel(col)
plt.title(f"Boxplot of {col} by heart_number and plate")
plt.tight_layout()
plt.show()


# In[ ]:




