#!/usr/bin/env python
# coding: utf-8

# # Split a selected plate into four data splits
# 
# For the selected plate, we generate four splits:
# 
# 1. **Female heart holdout**: all single cells from the female hearts (#32 and #64)
# 2. **Well holdout**: one random well per remaining heart number
# 3. **Training**: 70% of the remaining single cells
# 4. **Testing**: 30% of the remaining single cells
# 
# The training and testing splits are stratified by heart failure subtype (Healthy, DCM, and ICM) so that one class does not dominate either split when training a multi-class model to predict subtype.

# In[1]:


import os
import pathlib
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ## Set paths and variables

# In[2]:


# Set random state for the whole notebook to ensure reproducibility
random_state = 0
random.seed(random_state)

# Set the plate to split and train with the PLATE_TO_TRAIN environment variable (defaults to the subtyping plate)
plate_to_train = os.environ.get("PLATE_TO_TRAIN", "CARD-CelIns-CX7_260814100001")

# Heart numbers from female patients to hold out entirely (only held out if they are on the plate)
female_hearts = [32, 64]

# Column with the heart failure subtype and the label to give healthy hearts (which are NaN)
subtype_col = "Metadata_heart_failure_type"
healthy_label = "Healthy"

# Set the ratio of the test data to 30% (training data will be 70%)
test_ratio = 0.30

# Path to directory with feature selected profiles
path_to_feature_selected_data = pathlib.Path(
    "../3.preprocessing_profiles/data/single_cell_profiles/"
).resolve(strict=True)

# Find feature selected parquet file (QC applied) for the selected plate (can be in a subfolder, e.g. hlhs/)
feature_selected_path = next(
    path_to_feature_selected_data.rglob(f"{plate_to_train}_sc_feature_selected.parquet")
)

# Make directory for split data (one folder per plate)
output_dir = pathlib.Path(f"./data_splits/{plate_to_train}")
output_dir.mkdir(parents=True, exist_ok=True)

# Print out the file found
print(f"Found feature selected file: {feature_selected_path.name}")


# ## Load in feature selected data

# In[3]:


# Load the feature selected file as a DataFrame
feature_selected_df = pd.read_parquet(feature_selected_path)

# Confirm that only the selected plate is in the data
assert feature_selected_df["Metadata_Plate"].nunique() == 1, (
    "More than one plate found in the feature selected file"
)

print(f"Loaded file: {feature_selected_path.name}")
print(f"Plate name: {plate_to_train}")
print(f"Shape: {feature_selected_df.shape}")
feature_selected_df.head()


# ## Set the heart failure subtype label used for stratification
# 
# Healthy hearts do not have a heart failure type (NaN), so we label them as `Healthy` to stratify across all three classes (Healthy, DCM, and ICM).
# This label is only used for stratifying and is not saved to the splits.

# In[4]:


# Label healthy hearts (NaN heart failure type) to get the three subtype classes
subtype_labels = feature_selected_df[subtype_col].fillna(healthy_label)

# Print the hearts and number of single cells per subtype
print(
    feature_selected_df.assign(subtype=subtype_labels)
    .groupby("subtype")
    .agg(
        hearts=("Metadata_heart_number", lambda x: sorted(x.unique())),
        num_cells=("Metadata_heart_number", "size"),
    )
)


# ## Split #1: Hold out the female hearts
# 
# Hold out all single cells from the female hearts (#32 and #64).

# In[5]:


# Hold out all rows from the female hearts (heart numbers are compared as strings since some plates use letters)
female_holdout_mask = (
    feature_selected_df["Metadata_heart_number"].astype(str).isin(map(str, female_hearts))
)
female_holdout_df = feature_selected_df[female_holdout_mask].copy()

# Save female heart holdout set (not made if the plate does not have the female hearts, e.g. the HLHS plate)
if female_holdout_df.empty:
    print(f"No female hearts {female_hearts} on {plate_to_train}, so no female heart holdout set is made")
else:
    female_holdout_df.to_parquet(output_dir / "holdout_female_hearts_split.parquet")

print(f"Female heart holdout set shape: {female_holdout_df.shape}")
print(female_holdout_df.groupby("Metadata_heart_number")["Metadata_Well"].nunique())


# ## Split #2: Hold out one random well per heart
# 
# Randomly hold out one well from each of the remaining hearts.

# In[6]:


# Randomly hold out one well per remaining heart (make sampling reproducible)
random_wells = (
    feature_selected_df[~female_holdout_mask]
    .drop_duplicates(subset=["Metadata_heart_number", "Metadata_Well"])
    .sort_values(["Metadata_heart_number", "Metadata_Well"])
    .groupby("Metadata_heart_number")["Metadata_Well"]
    .sample(1, random_state=random_state)
)
print(f"Randomly selected wells for holdout: {random_wells.tolist()}")

# Hold out all rows from the randomly selected wells
well_holdout_mask = ~female_holdout_mask & feature_selected_df["Metadata_Well"].isin(
    random_wells
)
well_holdout_df = feature_selected_df[well_holdout_mask].copy()

# Save well holdout set
well_holdout_df.to_parquet(output_dir / "holdout_well_split.parquet")

print(f"Well holdout set shape: {well_holdout_df.shape}")
print(well_holdout_df.groupby("Metadata_heart_number")["Metadata_Well"].unique())


# ## Splits #3 and #4: Training and testing (70/30)
# 
# Split the remaining single cells into training and testing sets, stratified by heart failure subtype.

# In[7]:


# Remaining data after removing both holdout sets
model_df = feature_selected_df[~female_holdout_mask & ~well_holdout_mask].copy()
print(f"Data shape (after dropping holdout rows): {model_df.shape}")

# Split data into training and test sets stratified by subtype
train_df, test_df = train_test_split(
    model_df,
    test_size=test_ratio,
    stratify=subtype_labels.loc[model_df.index],
    random_state=random_state,
)

# Save training and test data
train_df.to_parquet(output_dir / "training_split.parquet")
test_df.to_parquet(output_dir / "testing_split.parquet")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")


# ## Check the splits

# In[8]:


splits = {
    **({"holdout_female_hearts": female_holdout_df} if not female_holdout_df.empty else {}),
    "holdout_well": well_holdout_df,
    "training": train_df,
    "testing": test_df,
}

# Sanity check that every single cell is in exactly one split
all_split_idx = pd.concat(splits.values()).index
assert all_split_idx.is_unique, "Single cells found in more than one split"
assert (
    len(all_split_idx) == feature_selected_df.shape[0]
), "Splits do not add up to the original data"

# Sanity check that the female hearts and holdout wells are not in training or testing
for split_name in ["training", "testing"]:
    split_df = splits[split_name]
    assert not split_df.index.isin(female_holdout_df.index).any()
    assert not split_df["Metadata_Well"].isin(random_wells).any()

# Number of single cells per subtype in each split
subtype_counts = pd.DataFrame(
    {
        split_name: subtype_labels.loc[split_df.index].value_counts()
        for split_name, split_df in splits.items()
    }
).fillna(0).astype(int)
print("Number of single cells per subtype in each split:")
print(subtype_counts)

# Proportion of each subtype within each split
print("\nProportion of each subtype in each split:")
print((subtype_counts / subtype_counts.sum()).round(3))

