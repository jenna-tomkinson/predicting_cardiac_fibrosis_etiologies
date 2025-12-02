#!/usr/bin/env python
# coding: utf-8

# # Split DMSO plate into data splits for three different models
# 
# 1. Healthy versus failing (IDC) hearts; whole plate
# 2. Healthy versus failing (DCM subtype) hearts; filter plate
# 3. Healthy versus failing (HCM subtype) hearts; filter plate

# In[1]:


import pathlib
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ## Set paths and variables

# In[2]:


# Set random state for the whole notebook to ensure reproducibility
random_state = 0
random.seed(random_state)

# Path to directory with feature selected profiles
path_to_feature_selected_data = pathlib.Path(
    "../3.preprocessing_profiles/data/single_cell_profiles/"
).resolve(strict=True)

# Find feature selected parquet file (QC applied)
feature_selected_files = list(
    path_to_feature_selected_data.glob("*_feature_selected.parquet")
)

# Make directory for split data
output_dir = pathlib.Path("./data_splits")
output_dir.mkdir(exist_ok=True)

# Print out the files found
print(f"Found {len(feature_selected_files)} feature selected files:")
for file in feature_selected_files:
    print(f"- {file.name}")


# ## Load in feature selected data

# In[3]:


# Load the feature selected file as a DataFrame
feature_selected_file = feature_selected_files[0]
plate = pathlib.Path(feature_selected_file).stem.split("_")[0]
feature_selected_df = pd.read_parquet(feature_selected_file)

print(f"Loaded file: {feature_selected_file.name}")
print(f"Plate name: {plate}")
print(f"Shape: {feature_selected_df.shape}")
feature_selected_df.head()


# ## Drop all rows from heart #47 (due to over-confluence)

# In[4]:


# Drop rows from heart #47 (due to over-confluence)
feature_selected_df = feature_selected_df[
    feature_selected_df["Metadata_heart_number"] != 47
]
print(f"Shape after dropping Heart #47: {feature_selected_df.shape}")


# ## Perform splits for model #1 (healthy versus failing IDC)
# 
# Holdout all of heart #2 (media) and one random well from each heart from the model.

# In[5]:


# Set output directory for this model
model_output_dir = output_dir / "model_all_hearts"
model_output_dir.mkdir(parents=True, exist_ok=True)


# In[6]:


# Hold out all rows where treatment is None (only applies to heart 2)
holdout_mask = feature_selected_df["Metadata_treatment"] == "None"

# Randomly hold out one well per other heart (make sampling reproducible)
random_wells = (
    feature_selected_df[~holdout_mask]
    .groupby("Metadata_heart_number")["Metadata_Well"]
    .apply(lambda x: x.dropna().sample(1, random_state=random_state))
    .explode()
)
print(f"Randomly selected wells for holdout: {random_wells.tolist()}")

# Combine with heart 2 / 'None' treatment rows
holdout_idx = feature_selected_df[holdout_mask].index.union(
    feature_selected_df.index[feature_selected_df["Metadata_Well"].isin(random_wells)]
)

# Create holdout and remaining dataframes
holdout_df = feature_selected_df.loc[holdout_idx].copy()
model_1_df = feature_selected_df.drop(holdout_idx).copy()

# Save holdout set for model_1
holdout_df.to_parquet(model_output_dir / "holdout_split.parquet")

print(f"Holdout set shape: {holdout_df.shape}")


# In[7]:


print(f"Model 1 data shape (after dropping holdout rows): {model_1_df.shape}")

# Sanity check
assert (
    holdout_df.shape[0] + model_1_df.shape[0] == feature_selected_df.shape[0]
), "Holdout + remaining does not equal original after splitting"

# Set the ratio of the test data to 30% (training data will be 70%)
test_ratio = 0.30

# Split data into training and test sets
train_df, test_df = train_test_split(
    model_1_df,
    test_size=test_ratio,
    stratify=model_1_df["Metadata_cell_type"],
    random_state=random_state,
)

# Save training and test data
train_df.to_parquet(model_output_dir / "training_split.parquet")
test_df.to_parquet(model_output_dir / "testing_split.parquet")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Print out the number of samples per cell type in each split
print("\nCell type distribution in training set:")
print(train_df["Metadata_cell_type"].value_counts(normalize=True))


# ## Perform splits for model #2 (healthy versus failing DCM subtype)
# 
# Filter for only healthy and failing hearts from DCM patients. Holdout all of heart #2 (media) and one random well from each heart from the model.

# In[8]:


# Set output directory for this model
model_output_dir = output_dir / "model_DCM"
model_output_dir.mkdir(parents=True, exist_ok=True)


# In[9]:


# Filter the feature_selected_df for healthy hearts and failing hearts with DCM
filtered_df = feature_selected_df[
    (feature_selected_df["Metadata_cell_type"] == "Healthy")
    | (
        (feature_selected_df["Metadata_cell_type"] == "Failing")
        & (feature_selected_df["Metadata_heart_failure_type"] == "DCM")
    )
].copy()

print(f"Filtered DataFrame shape (Healthy + Failing DCM): {filtered_df.shape}")
print(
    "Unique heart numbers in filtered DataFrame:",
    filtered_df["Metadata_heart_number"].unique(),
)
filtered_df.head()


# In[10]:


# Hold out all rows where treatment is None (only applies to heart 2)
holdout_mask = filtered_df["Metadata_treatment"] == "None"

# Randomly hold out one well per other heart (make sampling reproducible)
random_wells = (
    filtered_df[~holdout_mask]
    .groupby("Metadata_heart_number")["Metadata_Well"]
    .apply(lambda x: x.dropna().sample(1, random_state=random_state))
    .explode()
)
print(f"Randomly selected wells for holdout: {random_wells.tolist()}")

# Combine with heart 2 / 'None' treatment rows
holdout_idx = filtered_df[holdout_mask].index.union(
    filtered_df.index[filtered_df["Metadata_Well"].isin(random_wells)]
)

# Create holdout and remaining dataframes
holdout_df = filtered_df.loc[holdout_idx].copy()
model_2_df = filtered_df.drop(holdout_idx).copy()

# Save holdout set for model_2
holdout_df.to_parquet(model_output_dir / "holdout_split.parquet")

print(f"Holdout set shape: {holdout_df.shape}")


# In[11]:


print(f"Model 1 data shape (after dropping holdout rows): {model_2_df.shape}")

# Sanity check
assert (
    holdout_df.shape[0] + model_2_df.shape[0] == filtered_df.shape[0]
), "Holdout + remaining does not equal original after splitting"

# Set the ratio of the test data to 30% (training data will be 70%)
test_ratio = 0.30

# Split data into training and test sets
train_df, test_df = train_test_split(
    model_2_df,
    test_size=test_ratio,
    stratify=model_2_df["Metadata_cell_type"],
    random_state=random_state,
)

# Save training and test data
train_df.to_parquet(model_output_dir / "training_split.parquet")
test_df.to_parquet(model_output_dir / "testing_split.parquet")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")


# ## Perform splits for model #3 (healthy versus failing HCM subtype)
# 
# Filter for only healthy and failing hearts from HCM patients. Holdout all of heart #2 (media) and one random well from each heart from the model.

# In[12]:


# Set output directory for this model
model_output_dir = output_dir / "model_HCM"
model_output_dir.mkdir(parents=True, exist_ok=True)


# In[13]:


# Filter the feature_selected_df for healthy hearts and failing hearts with HCM
filtered_df = feature_selected_df[
    (feature_selected_df["Metadata_cell_type"] == "Healthy")
    | (
        (feature_selected_df["Metadata_cell_type"] == "Failing")
        & (feature_selected_df["Metadata_heart_failure_type"] == "HCM")
    )
].copy()

print(f"Filtered DataFrame shape (Healthy + Failing HCM): {filtered_df.shape}")
print(
    "Unique heart numbers in filtered DataFrame:",
    filtered_df["Metadata_heart_number"].unique(),
)
filtered_df.head()


# In[14]:


# Hold out all rows where treatment is None (only applies to heart 2)
holdout_mask = filtered_df["Metadata_treatment"] == "None"

# Randomly hold out one well per other heart (make sampling reproducible)
random_wells = (
    filtered_df[~holdout_mask]
    .groupby("Metadata_heart_number")["Metadata_Well"]
    .apply(lambda x: x.dropna().sample(1, random_state=random_state))
    .explode()
)
print(f"Randomly selected wells for holdout: {random_wells.tolist()}")

# Combine with heart 2 / 'None' treatment rows
holdout_idx = filtered_df[holdout_mask].index.union(
    filtered_df.index[filtered_df["Metadata_Well"].isin(random_wells)]
)

# Create holdout and remaining dataframes
holdout_df = filtered_df.loc[holdout_idx].copy()
model_3_df = filtered_df.drop(holdout_idx).copy()

# Save holdout set for model_3
holdout_df.to_parquet(model_output_dir / "holdout_split.parquet")

print(f"Holdout set shape: {holdout_df.shape}")


# In[15]:


print(f"Model 1 data shape (after dropping holdout rows): {model_3_df.shape}")

# Sanity check
assert (
    holdout_df.shape[0] + model_3_df.shape[0] == filtered_df.shape[0]
), "Holdout + remaining does not equal original after splitting"

# Set the ratio of the test data to 30% (training data will be 70%)
test_ratio = 0.30

# Split data into training and test sets
train_df, test_df = train_test_split(
    model_3_df,
    test_size=test_ratio,
    stratify=model_3_df["Metadata_cell_type"],
    random_state=random_state,
)

# Save training and test data
train_df.to_parquet(model_output_dir / "training_split.parquet")
test_df.to_parquet(model_output_dir / "testing_split.parquet")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

