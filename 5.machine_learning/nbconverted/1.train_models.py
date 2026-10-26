#!/usr/bin/env python
# coding: utf-8

# # Train binary and multi-class models with subsequent shuffled baselines
# 
# Using the training split from the selected plate, we train:
# 
# 1. **Binary models** (healthy versus failing): all hearts together and one model per failing subtype (healthy versus that subtype only). Training data are downsampled to the smallest class.
# 2. **Multi-class model**: predicts heart failure subtype across all hearts. Training data are not downsampled.
# 
# Each model is trained with a shuffled baseline.

# In[1]:


import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend

sys.path.append("../utils")
from training_utils import downsample_data, get_heart_grouped_folds, get_X_y_data


# ## Helper functions

# In[ ]:


def print_convergence(model: LogisticRegression) -> None:
    """Print if the final (refit) model converged before reaching the max number of iterations.

    Args:
        model (LogisticRegression): best estimator from the random search
    """
    n_iter = int(model.n_iter_.max())
    status = "converged" if n_iter < model.max_iter else "DID NOT CONVERGE"
    print(f"Final fit {status}: {n_iter} of {model.max_iter} max iterations")


# In[2]:


# Set numpy seed for reproducibility
np.random.seed(0)

# Set the plate to train with the PLATE_TO_TRAIN environment variable (defaults to the subtyping plate)
plate_to_train = os.environ.get("PLATE_TO_TRAIN", "CARD-CelIns-CX7_260814100001")

# Metadata column used for prediction class
label = "Metadata_cell_type"

# Column with the heart failure subtype and the label to give healthy hearts (which are NaN)
subtype_col = "Metadata_heart_failure_type"
healthy_label = "Healthy"

# Metadata column with the heart each single cell comes from (used to group the cross validation folds)
group_col = "Metadata_heart_number"

# Number of cross validation folds (there are only 3 hearts for each failing subtype)
n_folds = 3

# Path to training/testing datasets for the selected plate
training_data_path = pathlib.Path(f"./data_splits/{plate_to_train}").resolve(
    strict=True
)

# Directories for outputs (one folder per plate)
model_dir = pathlib.Path(f"./models/{plate_to_train}")
model_dir.mkdir(exist_ok=True, parents=True)

encoder_dir = pathlib.Path(f"./encoder_results/{plate_to_train}")
encoder_dir.mkdir(exist_ok=True, parents=True)

# Load in the training split for the selected plate
train_df = pd.read_parquet(training_data_path / "training_split.parquet")

print(f"Plate name: {plate_to_train}")
print(f"Training data shape: {train_df.shape}")


# ## Set the training dataset for each binary model
# 
# All binary models predict healthy versus failing.
# The `model_all_hearts` dataset uses the whole training split and each subtype dataset is filtered to the healthy hearts and the failing hearts from that subtype.

# In[3]:


# Failing subtypes in the training data (healthy hearts do not have a subtype)
failing_subtypes = sorted(train_df[subtype_col].dropna().unique())

# Dictionary to store the training dataset for each binary model
training_dfs = {"model_all_hearts": train_df}

# Filter for only healthy hearts and failing hearts from each subtype
for subtype in failing_subtypes:
    training_dfs[f"model_{subtype}"] = train_df[
        (train_df[label] == healthy_label) | (train_df[subtype_col] == subtype)
    ].copy()

# Print the number of single cells per class before downsampling
for dataset_name, dataset_df in training_dfs.items():
    print(f"Dataset: {dataset_name} {dataset_df.shape}")
    print(dataset_df[label].value_counts(), "\n")


# In[4]:


# Loop through and downsample each training dataset
for dataset_name, dataset_df in training_dfs.items():
    # Downsample to the smallest class
    downsample_df = downsample_data(data=dataset_df, label=label)

    # Replace and store the downsampled dataframe
    training_dfs[dataset_name] = downsample_df

    # Export as a new parquet with just the rows after downsampling per model (not indices)
    output_file = (
        training_data_path / dataset_name / "downsample_training_split.parquet"
    )
    output_file.parent.mkdir(exist_ok=True, parents=True)
    downsample_df.to_parquet(output_file, index=False)

    print(f"Parquet file created at {output_file} with {downsample_df.shape[0]} rows.")
    print(downsample_df.shape)
    print(downsample_df[label].value_counts())


# In[5]:


# Collect all unique labels across all datasets
all_labels = set()
for dataset_name, dataset_df in training_dfs.items():
    all_labels.update(dataset_df[label].unique())

# Fit the LabelEncoder on the combined set of all labels
le = LabelEncoder()
le.fit(list(all_labels))

# Save the global label encoder for consistency
dump(le, encoder_dir / "label_encoder_global.joblib")

# Print the global class mapping
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Global Class Mapping:")
print(class_mapping)

# New dictionary to hold final training data with X and y
training_data = {}

# Process each dataset to get X and y
for dataset_name, dataset_df in training_dfs.items():
    # Non-shuffled data
    X_train, y_train = get_X_y_data(df=dataset_df, label=label, shuffle_features=False)
    y_train_encoded = le.transform(y_train)

    # Shuffled data
    X_shuffled_train, y_shuffled_train = get_X_y_data(
        df=dataset_df, label=label, shuffle_features=True
    )
    y_shuffled_train_encoded = le.transform(y_shuffled_train)

    # Store X, y, and the cross validation folds (grouped by heart) in the dictionary
    training_data[dataset_name] = {
        "X_train": X_train,
        "y_train": y_train_encoded,
        "X_shuffled_train": X_shuffled_train,
        "y_shuffled_train": y_shuffled_train_encoded,
        "cv_folds": get_heart_grouped_folds(
            df=dataset_df, group_col=group_col, strata_col=subtype_col, n_splits=n_folds
        ),
    }

    # Print the hearts in the validation data of each fold
    for fold, (_, val_idx) in enumerate(training_data[dataset_name]["cv_folds"]):
        val_hearts = sorted(dataset_df[group_col].iloc[val_idx].unique())
        print(f"{dataset_name} | fold {fold} validation hearts: {val_hearts}")


# In[6]:


# NOTE: Cross validation folds are grouped by heart (set per dataset above), so that all single
# cells from a heart are in the same fold and hyperparameters are selected for performance on
# hearts not trained on. Each fold holds out at least one heart per subtype.

# Set Logistic Regression model parameters
logreg_params = {
    "penalty": "elasticnet",
    "solver": "saga",
    "max_iter": 5000,
    "n_jobs": -1,
    "random_state": 0,
    "class_weight": "balanced",
}

# Define the hyperparameter search space for RandomizedSearchCV
param_dist = {
    "C": np.logspace(-2, 1, 7),  # values from 0.01 to 10
    "l1_ratio": np.linspace(0, 1, 11),
}

# Set the random search hyperparameterization method parameters
random_search_params = {
    "param_distributions": param_dist,
    "scoring": "f1_weighted",
    "random_state": 0,
    "n_jobs": -1,
}


# ## Train binary logistic regressions

# In[7]:


# Initialize Logistic Regression and RandomizedSearchCV
logreg = LogisticRegression(**logreg_params)

# Loop through the training data dictionary
for dataset_name, data_dict in training_data.items():
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_shuffled_train = data_dict["X_shuffled_train"]
    y_shuffled_train = data_dict["y_shuffled_train"]

    # Set the random search with the folds grouped by heart for this dataset
    random_search = RandomizedSearchCV(
        logreg, cv=data_dict["cv_folds"], **random_search_params
    )

    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )

            ########################################################
            # Train model on non-shuffled (final) training data
            ########################################################
            print(f"Training model for {dataset_name} (final)...")
            final_random_search = clone(random_search)
            final_random_search.fit(X_train, y_train)
            print(
                f"Optimal parameters for {dataset_name} (final):",
                final_random_search.best_params_,
            )
            print_convergence(final_random_search.best_estimator_)

            # Save model
            final_model_filename = model_dir / f"{dataset_name}_final_downsample.joblib"
            dump(final_random_search.best_estimator_, final_model_filename)
            print(f"Model saved as: {final_model_filename}")

            ########################################################
            # Train model on shuffled training data
            ########################################################
            print(f"Training model for {dataset_name} (shuffled)...")
            shuffled_random_search = clone(random_search)
            shuffled_random_search.fit(X_shuffled_train, y_shuffled_train)
            print(
                f"Optimal parameters for {dataset_name} (shuffled):",
                shuffled_random_search.best_params_,
            )
            print_convergence(shuffled_random_search.best_estimator_)

            # Save model
            shuffled_final_model_filename = (
                model_dir / f"{dataset_name}_shuffled_downsample.joblib"
            )
            dump(shuffled_random_search.best_estimator_, shuffled_final_model_filename)
            print(f"Model saved as: {shuffled_final_model_filename}")


# ## For all hearts, train a multi-class logistic regression to predict the Metadata_heart_failure_type
# 
# The multi-class model uses the whole training split with no downsampling (class weights are balanced in the model instead).

# In[8]:


# Update label for the model
label = subtype_col

# Use the whole training split (all hearts) with no downsampling
all_hearts_df = train_df.copy()

# Update the Metadata_heart_failure_type for NaNs to 'Healthy'
all_hearts_df[label] = all_hearts_df[label].fillna(healthy_label)

# Print shape and number of single cells per class in the training data
print(all_hearts_df.shape)
print(all_hearts_df[label].value_counts())


# In[9]:


# Fit LabelEncoder
le = LabelEncoder()
le.fit(all_hearts_df[label].unique())
dump(le, encoder_dir / "label_encoder_multi-class.joblib")

# Show class mapping
print("Multi-Class Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Get non-shuffled data
X_train, y_train = get_X_y_data(df=all_hearts_df, label=label, shuffle_features=False)
y_train_encoded = le.transform(y_train)

# Get shuffled data
X_shuffled_train, y_shuffled_train = get_X_y_data(
    df=all_hearts_df, label=label, shuffle_features=True
)
y_shuffled_train_encoded = le.transform(y_shuffled_train)

# Store in a single dictionary
multi_class_training_data = {
    "X_train": X_train,
    "y_train": y_train_encoded,
    "X_shuffled_train": X_shuffled_train,
    "y_shuffled_train": y_shuffled_train_encoded,
    "cv_folds": get_heart_grouped_folds(
        df=all_hearts_df, group_col=group_col, strata_col=label, n_splits=n_folds
    ),
}


# In[10]:


# Update just logreg_params for multi-class classification
logreg_params.update({"multi_class": "multinomial"})

# Print to confirm change
print("Updated Logistic Regression parameters for multi-class:", logreg_params)


# In[11]:


# Initialize Logistic Regression and RandomizedSearchCV
logreg = LogisticRegression(**logreg_params)
random_search = RandomizedSearchCV(
    logreg, cv=multi_class_training_data["cv_folds"], **random_search_params
)

# Extract data for the multi-class all_hearts model
X_train = multi_class_training_data["X_train"]
y_train = multi_class_training_data["y_train"]
X_shuffled_train = multi_class_training_data["X_shuffled_train"]
y_shuffled_train = multi_class_training_data["y_shuffled_train"]

with parallel_backend("multiprocessing"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

        ########################################################
        # Train model on non-shuffled (final) training data
        ########################################################
        print("Training multi-class model for all_hearts (final)...")
        final_random_search = clone(random_search)
        final_random_search.fit(X_train, y_train)
        print("Optimal parameters (final):", final_random_search.best_params_)
        print_convergence(final_random_search.best_estimator_)

        # Save model
        final_model_filename = model_dir / "model_all_hearts_final_multiclass.joblib"
        dump(final_random_search.best_estimator_, final_model_filename)
        print(f"Model saved as: {final_model_filename}")

        ########################################################
        # Train model on shuffled training data
        ########################################################
        print("Training multi-class model for all_hearts (shuffled)...")
        shuffled_random_search = clone(random_search)
        shuffled_random_search.fit(X_shuffled_train, y_shuffled_train)
        print("Optimal parameters (shuffled):", shuffled_random_search.best_params_)
        print_convergence(shuffled_random_search.best_estimator_)

        # Save model
        shuffled_final_model_filename = (
            model_dir / "model_all_hearts_shuffled_multiclass.joblib"
        )
        dump(shuffled_random_search.best_estimator_, shuffled_final_model_filename)
        print(f"Model saved as: {shuffled_final_model_filename}")

