#!/usr/bin/env python
# coding: utf-8

# # Check the cell counts across hearts

# In[1]:


import pandas as pd
import pathlib
import matplotlib.pyplot as plt
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


# In[3]:


# split heart 2 by treatment and plot and add to profile_df
profile_df["heart_treatment"] = profile_df["Metadata_heart_number"].astype(str)
mask = profile_df["Metadata_heart_number"] == 2
profile_df.loc[mask, "heart_treatment"] = (
    profile_df.loc[mask, "Metadata_heart_number"].astype(str)
    + "_"
    + profile_df.loc[mask, "Metadata_treatment"].fillna("None").astype(str)
)
print("Profile DataFrame shape:", profile_df.shape)


# In[4]:


# Compute counts per heart and plate
counts_df = (
    profile_df.groupby(["heart_treatment", "Metadata_Plate"])
    .size()
    .reset_index(name="cell_count")
)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=counts_df,
    x="heart_treatment",
    y="cell_count",
    hue="Metadata_Plate",
    palette="viridis",
)
plt.xticks(rotation=45)
plt.xlabel("Heart number")
plt.ylabel("Cell count")
plt.title("Cell counts per heart")
plt.tight_layout()
# Save plot
output_plot_dir = pathlib.Path("../4.EDA/figures")
output_plot_dir.mkdir(parents=True, exist_ok=True)
output_plot_path = output_plot_dir / "cell_counts_per_heart_treatment.png"
plt.savefig(output_plot_path)
print("Plot saved to:", output_plot_path)
plt.show()


# In[5]:


# Define the column to plot
col = "Cells_Neighbors_NumberOfNeighbors_Adjacent"

# Determine order automatically from the data
order = sorted(profile_df["heart_treatment"].dropna().unique())

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=profile_df.dropna(subset=[col]),
    x="heart_treatment",
    y=col,
    hue="Metadata_Plate",
    order=order,
    showcaps=True,
    fliersize=2,
    boxprops={"alpha": 0.8},
)

plt.xlabel("Heart Treatment")
plt.ylabel(col)
plt.title(f"Boxplot of {col} by heart_treatment and plate")
plt.tight_layout()
plt.show()

