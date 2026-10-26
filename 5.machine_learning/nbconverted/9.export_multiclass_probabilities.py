#!/usr/bin/env python
# coding: utf-8

# # Export the multi-class predicted probabilities per single cell
# 
# We apply the multi-class model (`model_all_hearts_final_multiclass`) to every single cell on the training plate and export the predicted probability of each class per cell.
# Each cell has the identifying metadata (plate, well, site, and nuclei object number) so the probabilities can be merged with the normalized single-cell profiles for the plate.
# 
# The `dataset` column shows the split each cell is from.
# Cells in the `train` split were used to train the model, so their probabilities are higher confidence than the cells the model has not seen (`test` and holdout splits).

# ## Import libraries

# In[1]:


import os
import pathlib
import sys

import pandas as pd
from IPython.display import display
from joblib import load

sys.path.append("../utils")
from training_utils import get_X_y_data


# ## Set paths and variables

# In[2]:


# Set the plate to process with the PLATE_TO_TRAIN environment variable (defaults to the subtyping plate)
plate_to_process = os.environ.get("PLATE_TO_TRAIN", "CARD-CelIns-CX7_260814100001")

model_path = pathlib.Path(
    f"./models/{plate_to_process}/model_all_hearts_final_multiclass.joblib"
).resolve(strict=True)
encoder_path = pathlib.Path(
    f"./encoder_results/{plate_to_process}/label_encoder_multi-class.joblib"
).resolve(strict=True)
split_dir = pathlib.Path(f"./data_splits/{plate_to_process}").resolve(strict=True)

# Normalized profiles for the plate (can be in a subfolder, e.g. hlhs/), used to check the cells match
normalized_path = next(
    pathlib.Path("../3.preprocessing_profiles/data/single_cell_profiles")
    .resolve(strict=True)
    .rglob(f"{plate_to_process}_sc_normalized.parquet")
)

# Output dir for the predicted probabilities
performance_metrics_dir = pathlib.Path(f"./performance_metrics/{plate_to_process}")
performance_metrics_dir.mkdir(parents=True, exist_ok=True)

# All splits (the female heart holdout is only made when the plate has the female hearts)
split_files = {
    "train": "training_split.parquet",
    "test": "testing_split.parquet",
    "holdout_well": "holdout_well_split.parquet",
    "holdout_female_hearts": "holdout_female_hearts_split.parquet",
}
split_files = {
    split_name: file_name
    for split_name, file_name in split_files.items()
    if (split_dir / file_name).exists()
}

# Columns that uniquely identify a single cell
cell_id_cols = [
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Nuclei_Number_Object_Number",
]
# Metadata to keep with the predictions
metadata_cols = cell_id_cols + [
    "Metadata_heart_number",
    "Metadata_cell_type",
    "Metadata_heart_failure_type",
    "Metadata_treatment",
]

# Healthy hearts have no heart failure type
label = "Metadata_heart_failure_type"
healthy_label = "Healthy"


# ## Predict the class probabilities for every single cell
# 
# The model classes are encoded, so we use the multi-class label encoder to get the class names.

# In[3]:


model = load(model_path)
label_encoder = load(encoder_path)
class_names = label_encoder.inverse_transform(model.classes_)

cells_df = pd.concat(
    [
        pd.read_parquet(split_dir / file_name).assign(dataset=split_name)
        for split_name, file_name in split_files.items()
    ],
    ignore_index=True,
)

# Class of each cell (healthy hearts have no heart failure type, so they are labeled as Healthy)
true_class = cells_df[label].fillna(healthy_label)

X, _ = get_X_y_data(df=cells_df, label=label, shuffle_features=False)
probabilities = model.predict_proba(X[model.feature_names_in_])

probabilities_df = pd.concat(
    [
        cells_df[metadata_cols + ["dataset"]],
        pd.DataFrame(
            {
                "true_class": true_class,
                "predicted_class": class_names[probabilities.argmax(axis=1)],
            }
        ),
        pd.DataFrame(probabilities, columns=[f"probability_{name}" for name in class_names]),
    ],
    axis=1,
)

print(probabilities_df.shape)
probabilities_df.head()


# ## Check that every cell matches one cell in the normalized profiles
# 
# Each cell should be in one split only, and every cell in the normalized profiles should have a prediction.

# In[4]:


normalized_ids_df = pd.read_parquet(normalized_path, columns=cell_id_cols)

assert not probabilities_df.duplicated(subset=cell_id_cols).any()
merged_df = normalized_ids_df.merge(
    probabilities_df[cell_id_cols], on=cell_id_cols, how="outer", validate="one_to_one", indicator=True
)
assert (merged_df["_merge"] == "both").all()

pd.DataFrame(
    {
        "n_cells_normalized": [normalized_ids_df.shape[0]],
        "n_cells_with_predictions": [probabilities_df.shape[0]],
    }
)


# ## Summarize and save the predicted probabilities

# In[5]:


# Mean predicted probability per split and class
display(
    probabilities_df.groupby(["dataset", "true_class"])[
        [f"probability_{name}" for name in class_names]
    ]
    .mean()
    .round(3)
    .assign(n_cells=probabilities_df.groupby(["dataset", "true_class"]).size())
)

output_path = performance_metrics_dir / "multiclass_single_cell_predicted_probabilities.parquet"
probabilities_df.to_parquet(output_path, index=False)

pd.DataFrame({"output_file": [str(output_path)], "n_cells": [probabilities_df.shape[0]]})

