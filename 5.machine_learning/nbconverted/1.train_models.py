#!/usr/bin/env python
# coding: utf-8

# # Train three models with subsequent shuffled baselines

# In[1]:


import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend

sys.path.append("../utils")
from training_utils import downsample_data, get_X_y_data


# In[2]:


# Set numpy seed for reproducibility
np.random.seed(0)

# Set if processing redo plate (true) or original plate (false)
redo_plate = True

# Metadata column used for prediction class
label = "Metadata_cell_type"

if redo_plate:
    # Path to training/testing datasets for redo plate
    training_data_path = pathlib.Path("./data_splits/redo_DMSO_plate")
    # Directories for outputs
    model_dir = pathlib.Path("./models/redo_DMSO_plate")
    model_dir.mkdir(exist_ok=True, parents=True)

    encoder_dir = pathlib.Path("./encoder_results/redo_DMSO_plate")
    encoder_dir.mkdir(exist_ok=True, parents=True)
else:  # process original plate
    # Path to training/testing datasets for original plate
    training_data_path = pathlib.Path("./data_splits/original_DMSO_plate")
    # Directories for outputs
    model_dir = pathlib.Path("./models/original_DMSO_plate")
    model_dir.mkdir(exist_ok=True, parents=True)

    encoder_dir = pathlib.Path("./encoder_results/original_DMSO_plate")
    encoder_dir.mkdir(exist_ok=True, parents=True)

# Find all training datasets
training_files = list(training_data_path.rglob("training_split.parquet"))

print(f"Found {len(training_files)} training datasets.")

# Dictionary to store loaded training datasets
training_dfs = {}

# Loop through and load each training dataset
for training_file in training_files:
    dataset_name = training_file.parent.name  # Use parent folder name as key
    print(f"Loading dataset: {dataset_name}")  # only print the model/folder name

    train_df = pd.read_parquet(training_file)
    training_dfs[dataset_name] = train_df


# In[3]:


# Loop through and downsample each loaded training dataset
for dataset_name, train_df in training_dfs.items():
    # Downsample to the smallest class
    downsample_df = downsample_data(data=train_df, label=label)

    # Replace and store the downsampled dataframe
    training_dfs[dataset_name] = downsample_df

    # Export as a new parquet with just the rows after downsampling per model (not indices)
    output_file = (
        training_data_path / dataset_name / "downsample_training_split.parquet"
    )
    downsample_df.to_parquet(output_file, index=False)

    print(f"Parquet file created at {output_file} with {downsample_df.shape[0]} rows.")
    print(downsample_df.shape)
    print(downsample_df[label].value_counts())


# In[4]:


# Collect all unique labels across all datasets
all_labels = set()
for dataset_name, train_df in training_dfs.items():
    all_labels.update(train_df[label].unique())

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
for dataset_name, train_df in training_dfs.items():
    # Non-shuffled data
    X_train, y_train = get_X_y_data(df=train_df, label=label, shuffle_features=False)
    y_train_encoded = le.transform(y_train)

    # Shuffled data
    X_shuffled_train, y_shuffled_train = get_X_y_data(
        df=train_df, label=label, shuffle_features=True
    )
    y_shuffled_train_encoded = le.transform(y_shuffled_train)

    # Store X and y in the dictionary
    training_data[dataset_name] = {
        "X_train": X_train,
        "y_train": y_train_encoded,
        "X_shuffled_train": X_shuffled_train,
        "y_shuffled_train": y_shuffled_train_encoded,
    }


# In[5]:


# Set folds for k-fold cross validation (default is 5, shuffle=True)
straified_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Set Logistic Regression model parameters (use default for max_iter)
logreg_params = {
    "penalty": "elasticnet",
    "solver": "saga",
    "max_iter": 1000,
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
    "cv": straified_k_folds,
}


# ## Train binary logistic regressions

# In[6]:


# Initialize Logistic Regression and RandomizedSearchCV
logreg = LogisticRegression(**logreg_params)
random_search = RandomizedSearchCV(logreg, **random_search_params)

# Loop through the training data dictionary
for dataset_name, data_dict in training_data.items():
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_shuffled_train = data_dict["X_shuffled_train"]
    y_shuffled_train = data_dict["y_shuffled_train"]

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

            # Save model
            shuffled_final_model_filename = (
                model_dir / f"{dataset_name}_shuffled_downsample.joblib"
            )
            dump(shuffled_random_search.best_estimator_, shuffled_final_model_filename)
            print(f"Model saved as: {shuffled_final_model_filename}")


# ## For all_hearts only model, train a multi-class logistic regression to predict the Metadata_heart_failure_type

# In[7]:


# Update label for the model
label = "Metadata_heart_failure_type"

# Load in the all hearts dataset
all_hearts_file = training_data_path / "model_all_hearts" / "training_split.parquet"
all_hearts_df = pd.read_parquet(all_hearts_file)

# Update the Metadata_heart_failure_type for NaNs to 'Healthy'
all_hearts_df[label] = all_hearts_df[label].fillna("Healthy")

# Print shape and value counts for the updated column
print(all_hearts_df.shape)
print(all_hearts_df[label].value_counts())


# In[8]:


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
}


# In[9]:


# Update just logreg_params for multi-class classification
logreg_params.update({"multi_class": "multinomial"})

# Print to confirm change
print("Updated Logistic Regression parameters for multi-class:", logreg_params)


# In[10]:


# Initialize Logistic Regression and RandomizedSearchCV
logreg = LogisticRegression(**logreg_params)
random_search = RandomizedSearchCV(logreg, **random_search_params)

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

        # Save model
        shuffled_final_model_filename = (
            model_dir / "model_all_hearts_shuffled_multiclass.joblib"
        )
        dump(shuffled_random_search.best_estimator_, shuffled_final_model_filename)
        print(f"Model saved as: {shuffled_final_model_filename}")

