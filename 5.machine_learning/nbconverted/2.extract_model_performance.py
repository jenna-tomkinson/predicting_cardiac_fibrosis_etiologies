#!/usr/bin/env python
# coding: utf-8

# # Extract model performance metrics
# 
# In this notebook, we extract metrics to evaluate performance such as:
# 
# 1. Precision-recall
# 2. Predicted probabilities

# ## Import libraries

# In[1]:


import os
import pathlib
import sys

import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

sys.path.append("../utils")
from training_utils import get_X_y_data


# ## Helper functions

# In[2]:


def get_pr_curve_results(
    model: LogisticRegression, df: pd.DataFrame, label: str, label_encoder: LabelEncoder
) -> pd.DataFrame:
    """Collect the precision-recall curve results from a model and dataset.

    Args:
        model (LogisticRegression): loaded in logistic regression model to collect results from
        df (pd.DataFrame): dataframe containing the data to apply model to
        label (str): label with the class being predicted
        label_encoder (LabelEncoder): encoder to transform the labels to integers

    Returns:
        pd.DataFrame: dataframe with the PR curve results for that data and model
    """
    try:
        # Get X and y data for the model
        X, y = get_X_y_data(df=df, label=label, shuffle_features=False)

        assert all(
            col in model.feature_names_in_ for col in X
        ), "Features in the model do not match the columns in the dataset"

        # Transform labels
        y_encoded = label_encoder.transform(y)

        # Ensure binary problem (this helper is for binary models only)
        unique_labels = set(y_encoded)
        assert (
            len(unique_labels) == 2
        ), f"Expected binary labels after encoding, got classes: {sorted(list(unique_labels))}"

        # Ensure model supports predict_proba and returns expected shape
        if not hasattr(model, "predict_proba"):
            raise AssertionError(
                "Model does not implement predict_proba required for PR curve"
            )

        y_proba = model.predict_proba(X)
        if (
            not hasattr(y_proba, "shape")
            or len(y_proba.shape) != 2
            or y_proba.shape[1] < 2
        ):
            raise AssertionError(
                f"predict_proba returned unexpected shape: {getattr(y_proba, 'shape', None)}"
            )

        y_scores = y_proba[:, 1]

        assert len(y_scores) == len(
            y_encoded
        ), "Length mismatch between predicted scores and labels"

        precision, recall, _ = precision_recall_curve(y_encoded, y_scores)

        return pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
            }
        )

    except Exception as e:
        raise AssertionError(
            f"Failed to compute PR curve for label '{label}': {e}"
        ) from e


# In[3]:


def get_predicted_probabilities(
    model: LogisticRegression, df: pd.DataFrame, label: str, label_encoder: LabelEncoder
) -> pd.DataFrame:
    """Collect predicted probabilities per single-cell from the model and dataset.

    Args:
        model (LogisticRegression): loaded in logistic regression model to collect results from
        df (pd.DataFrame): dataframe containing the data to apply model to
        label (str): label with the class being predicted
        label_encoder (LabelEncoder): encoder to transform the labels to integers

    Returns:
        pd.DataFrame: dataframe with the predicted probabilities per single-cell
    """
    try:
        # Validate required metadata columns exist for output
        for col in ("Metadata_treatment", "Metadata_heart_number"):
            assert col in df.columns, f"Required column '{col}' not found in dataframe"

        # Set treatment and heart number metadata to include in output
        metadata_treatment = df["Metadata_treatment"].values
        metadata_heart_number = df["Metadata_heart_number"].values

        # Get X and y for the model
        X, y = get_X_y_data(df=df, label=label, shuffle_features=False)

        # Ensure model features match dataset columns
        assert all(
            col in model.feature_names_in_ for col in X
        ), "Features in the model do not match the columns in the dataset"

        # Encode labels and ensure binary problem (this helper is for binary models only)
        # NOTE: A data split can have only one of the two classes (e.g., female heart holdout)
        y_encoded = label_encoder.transform(y)
        unique_labels = set(y_encoded)
        assert (
            len(unique_labels) <= 2
        ), f"Expected binary labels after encoding, got classes: {sorted(list(unique_labels))}"

        # Ensure model supports predict_proba and returns expected shape
        if not hasattr(model, "predict_proba"):
            raise AssertionError(
                "Model does not implement predict_proba required to get predicted probabilities"
            )

        y_proba = model.predict_proba(X)
        if not hasattr(y_proba, "shape") or len(y_proba.shape) != 2:
            raise AssertionError(
                f"predict_proba returned unexpected shape: {getattr(y_proba, 'shape', None)}"
            )
        if y_proba.shape[1] < 2:
            raise AssertionError(
                f"predict_proba returned fewer than 2 class probabilities: shape {y_proba.shape}"
            )
        if y_proba.shape[0] != len(X):
            raise AssertionError(
                f"predict_proba returned {y_proba.shape[0]} rows but expected {len(X)}"
            )

        # Use probability for the positive class
        y_scores = y_proba[:, 1]

        # Consistency checks
        assert len(y_scores) == len(
            y_encoded
        ), "Length mismatch between predicted probabilities and labels"
        assert len(metadata_treatment) == len(y_scores) and len(
            metadata_heart_number
        ) == len(
            y_scores
        ), "Length mismatch between metadata columns and predicted probabilities"

        return pd.DataFrame(
            {
                "actual_label": y,
                "predicted_probability": y_scores,
                "Metadata_treatment": metadata_treatment,
                "Metadata_heart_number": metadata_heart_number,
            }
        )

    except Exception as e:
        raise AssertionError(
            f"Failed to compute predicted probabilities for label '{label}': {e}"
        ) from e


# In[4]:


def get_model_datasets(
    model_name: str,
    training_data_path: pathlib.Path,
    eval_dfs: dict[str, pd.DataFrame],
    label: str,
    subtype_col: str,
    healthy_label: str,
) -> dict[str, pd.DataFrame]:
    """Get the data splits to apply to a binary model.

    Args:
        model_name (str): name of the model (e.g., model_all_hearts or model_DCM)
        training_data_path (pathlib.Path): path to the (downsampled) training data for the model
        eval_dfs (dict[str, pd.DataFrame]): testing and holdout data splits for all hearts
        label (str): label with the class being predicted
        subtype_col (str): column with the heart failure subtype
        healthy_label (str): label for the healthy hearts in the label column

    Returns:
        dict[str, pd.DataFrame]: dictionary with the dataframe per data split
    """
    datasets = {"train": pd.read_parquet(training_data_path)}

    # Subtype models only use healthy hearts and failing hearts from that subtype
    subtype = model_name.replace("model_", "")
    for split_name, split_df in eval_dfs.items():
        if subtype != "all_hearts":
            split_df = split_df[
                (split_df[label] == healthy_label) | (split_df[subtype_col] == subtype)
            ]
        datasets[split_name] = split_df

    return datasets


# # Set paths

# In[5]:


# Set the plate to process with the PLATE_TO_TRAIN environment variable (defaults to the subtyping plate)
plate_to_process = os.environ.get("PLATE_TO_TRAIN", "CARD-CelIns-CX7_260814100001")

# Directory with the training, testing, and holdout datasets for the plate
data_dir = pathlib.Path(f"data_splits/{plate_to_process}").resolve(strict=True)

# Directory with the trained models
model_dir = pathlib.Path(f"models/{plate_to_process}").resolve(strict=True)

# Directory with encoder
encoder_dir = pathlib.Path(f"encoder_results/{plate_to_process}").resolve(strict=True)

# Output directory the performance metrics
performance_metrics_dir = pathlib.Path(f"performance_metrics/{plate_to_process}")
performance_metrics_dir.mkdir(exist_ok=True, parents=True)

# Label being predicted
label = "Metadata_cell_type"

# Column with the heart failure subtype and the label to give healthy hearts (which are NaN)
subtype_col = "Metadata_heart_failure_type"
healthy_label = "Healthy"

# Paths to the data splits not used for training (shared across all models)
eval_split_paths = {
    "test": pathlib.Path(data_dir / "testing_split.parquet").resolve(strict=True),
    "holdout_well": pathlib.Path(data_dir / "holdout_well_split.parquet").resolve(
        strict=True
    ),
}

# Female heart holdout is only made when the plate has the female hearts
if (data_dir / "holdout_female_hearts_split.parquet").exists():
    eval_split_paths["holdout_female_hearts"] = data_dir / "holdout_female_hearts_split.parquet"


# ## Create dictionary with all relevant paths per model to extract metrics
# 
# The testing and holdout splits are shared across all models.
# For the subtype binary models, these splits are filtered to the healthy hearts and the failing hearts from that subtype (same as the training data).

# In[6]:


# Extract model names from model filenames
model_names = sorted(
    f.stem.replace("_final_downsample", "")
    for f in model_dir.glob("*_final_downsample.joblib")
)

# Create a nested dictionary with info per model
models_dict = {}
for model in model_names:
    models_dict[model] = {
        "training_data": pathlib.Path(
            data_dir / model / "downsample_training_split.parquet"
        ).resolve(strict=True),
        "final_model": pathlib.Path(
            model_dir / f"{model}_final_downsample.joblib"
        ).resolve(strict=True),
        "shuffled_model": pathlib.Path(
            model_dir / f"{model}_shuffled_downsample.joblib"
        ).resolve(strict=True),
        "encoder_result": pathlib.Path(
            encoder_dir / "label_encoder_global.joblib"
        ).resolve(strict=True),
    }

# Print out dictionary keys and paths for verification
for model, paths in models_dict.items():
    lines = [f"Model: {model}"] + [f"  {key}: {path}" for key, path in paths.items()]
    print("\n".join(lines))

# Load in the testing and holdout splits
eval_dfs = {
    split_name: pd.read_parquet(split_path)
    for split_name, split_path in eval_split_paths.items()
}


# In[7]:


# For each model, print unique Metadata_heart_number and class counts per data split
for model_name in models_dict:
    print(f"Model: {model_name}")
    datasets = get_model_datasets(
        model_name=model_name,
        training_data_path=models_dict[model_name]["training_data"],
        eval_dfs=eval_dfs,
        label=label,
        subtype_col=subtype_col,
        healthy_label=healthy_label,
    )
    for split_name, df in datasets.items():
        # Print unique heart numbers horizontally, sorted and compact
        hearts = sorted(df["Metadata_heart_number"].dropna().unique().tolist())
        hearts_str = ", ".join(str(h) for h in hearts)
        print(
            f"  {split_name} ({len(hearts)}): {hearts_str} | {df[label].value_counts().to_dict()}"
        )


# In[8]:


# Compute number of single cells per held-out well for each heart
cells_per_well = (
    eval_dfs["holdout_well"]
    .groupby(["Metadata_heart_number", "Metadata_Well"])
    .size()
    .reset_index(name="n_cells")
)

# Print results
print("Held-out Metadata_Well and cell count per Metadata_heart_number:")
print(cells_per_well.to_string(index=False))


# ## Extract metrics from the data splits applied to their respective models

# In[9]:


# Initialize results list
test_train_pr_results = []
test_train_probability_results = []

# Run through each model and get the PR results
for model_name, paths in models_dict.items():
    # Load the models and data
    final_model = load(paths["final_model"])
    shuffled_model = load(paths["shuffled_model"])
    label_encoder = load(paths["encoder_result"])

    print(f"Processing model: {model_name}")

    # Set dictionary with the data splits
    datasets = get_model_datasets(
        model_name=model_name,
        training_data_path=models_dict[model_name]["training_data"],
        eval_dfs=eval_dfs,
        label=label,
        subtype_col=subtype_col,
        healthy_label=healthy_label,
    )

    # Loop through both datasets and models
    for dataset_name, dataset in datasets.items():
        for model_type, model in [("final", final_model), ("shuffled", shuffled_model)]:
            # Get per-sample predicted probabilities
            prob_df = get_predicted_probabilities(
                model=model, df=dataset, label=label, label_encoder=label_encoder
            )
            prob_df["model_type"] = model_type
            prob_df["dataset"] = dataset_name
            prob_df["model_name"] = model_name
            test_train_probability_results.append(prob_df)

            # PR curve can not be computed when the data split only has one class
            if dataset[label].nunique() < 2:
                print(
                    f"{model_name} | {model_type} | {dataset_name} → Done (no PR curve, only one class)"
                )
                continue

            # Get PR curve results (global)
            pr_df = get_pr_curve_results(
                model=model, df=dataset, label=label, label_encoder=label_encoder
            )
            pr_df["model_type"] = model_type
            pr_df["dataset"] = dataset_name
            pr_df["model_name"] = model_name
            test_train_pr_results.append(pr_df)

            print(f"{model_name} | {model_type} | {dataset_name} → Done")

# Combine all results into one dataframe
all_models_pr_results_df = pd.concat(test_train_pr_results, ignore_index=True)
all_models_probabilities_df = pd.concat(
    test_train_probability_results, ignore_index=True
)

# Save the results
all_models_pr_results_df.to_parquet(
    performance_metrics_dir / "all_models_pr_curve_results.parquet", index=False
)
all_models_probabilities_df.to_parquet(
    performance_metrics_dir / "all_models_predicted_probabilities.parquet", index=False
)

# Check output
print(all_models_pr_results_df.shape)
all_models_pr_results_df.head(2)


# ## Extract performance from the multi-class model only

# In[10]:


# Load in final and shuffled model for multi-class models
final_model = load(pathlib.Path(model_dir / "model_all_hearts_final_multiclass.joblib"))
shuffled_model = load(
    pathlib.Path(model_dir / "model_all_hearts_shuffled_multiclass.joblib")
)

# Set paths to data splits (did not downsample training data for the multi-class model)
multi_class_split_paths = {
    "train": pathlib.Path(data_dir / "training_split.parquet").resolve(strict=True),
    **eval_split_paths,
}


# In[11]:


# Initialize results list
multi_class_pr_results = []
heart_accuracy_results = []
cell_probs_results = []

# Load in label encoder for multi-class model
label_encoder = load(pathlib.Path(encoder_dir / "label_encoder_multi-class.joblib"))

# Save the mapping from the encoded class to the class name (used for plotting in R)
pd.DataFrame(
    {
        "encoded_label": label_encoder.transform(label_encoder.classes_),
        "class_label": label_encoder.classes_,
    }
).to_csv(performance_metrics_dir / "multi_class_label_mapping.csv", index=False)

# Set updated label variable
label = "Metadata_heart_failure_type"

for dataset_name, data_path in multi_class_split_paths.items():
    # Load dataset
    df = pd.read_parquet(data_path)
    df[label] = df[label].fillna(healthy_label)

    # Get X and y for the model
    X, y = get_X_y_data(df=df, label=label, shuffle_features=False)
    y_encoded = label_encoder.transform(y)

    for model_type, model in [("final", final_model), ("shuffled", shuffled_model)]:
        y_pred_proba = model.predict_proba(X)
        y_pred_label = y_pred_proba.argmax(axis=1)

        # --- Save predicted probabilities per cell ---
        cell_probs_results.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "model_type": model_type,
                    "model_name": "model_all_hearts_multiclass",
                    "Metadata_heart_number": df["Metadata_heart_number"],
                    "true_label": y_encoded,
                    "predicted_label": y_pred_label,
                    **{
                        f"proba_class_{i}": y_pred_proba[:, i]
                        for i in range(y_pred_proba.shape[1])
                    },
                }
            )
        )

        # --- Compute PR curve per class ---
        for class_index, class_label in enumerate(model.classes_):
            y_true_binary = (y_encoded == class_index).astype(int)
            y_scores = y_pred_proba[:, class_index]

            # PR curve is not informative when the data split only has one class
            if y_true_binary.min() == y_true_binary.max():
                print(
                    f"model_all_hearts_multiclass | {model_type} | {dataset_name} | {class_label} → Skipped PR curve (only one class in split)"
                )
                continue

            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)

            pr_df = pd.DataFrame(
                {
                    "precision": precision,
                    "recall": recall,
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "model_name": "model_all_hearts_multiclass",
                    "class_label": class_label,
                }
            )

            multi_class_pr_results.append(pr_df)

            print(
                f"model_all_hearts_multiclass | {model_type} | {dataset_name} | {class_label} → Done"
            )

        # --- Compute per-heart × treatment accuracy ---
        for (heart, treatment), group_df in df.groupby(
            ["Metadata_heart_number", "Metadata_treatment"]
        ):
            mask = X.index.isin(group_df.index)
            if not mask.any():
                continue

            heart_treatment_acc = (y_pred_label[mask] == y_encoded[mask]).mean()
            heart_accuracy_results.append(
                {
                    "dataset": dataset_name,
                    "model_type": model_type,
                    "model_name": "model_all_hearts_multiclass",
                    "heart_number": heart,
                    "treatment": treatment,
                    "accuracy": heart_treatment_acc,
                }
            )

# Save PR results
pr_results_df = pd.concat(multi_class_pr_results, ignore_index=True)
pr_results_df.to_parquet(
    performance_metrics_dir / "multi_class_pr_results.parquet", index=False
)

# Save heart accuracy
heart_accuracy_df = pd.DataFrame(heart_accuracy_results)
heart_accuracy_df.to_parquet(
    performance_metrics_dir / "multi_class_heart_accuracy.parquet", index=False
)

# Save per-cell predicted probabilities
cell_probs_df = pd.concat(cell_probs_results, ignore_index=True)
cell_probs_df.to_parquet(
    performance_metrics_dir / "multi_class_cell_probabilities.parquet", index=False
)

print(
    f"Saved PR results → {performance_metrics_dir / 'multi_class_pr_results.parquet'}"
)
print(
    f"Saved heart accuracy → {performance_metrics_dir / 'multi_class_heart_accuracy.parquet'}"
)
print(
    f"Saved cell probabilities → {performance_metrics_dir / 'multi_class_cell_probabilities.parquet'}"
)

