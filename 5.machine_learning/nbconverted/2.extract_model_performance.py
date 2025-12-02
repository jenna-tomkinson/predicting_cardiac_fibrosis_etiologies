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


import pathlib
import sys

import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve

sys.path.append("../utils")
from training_utils import get_X_y_data


# ## Helper function to collect precision-recall results and predicted probabilities for binary models only

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
        y_encoded = label_encoder.transform(y)
        unique_labels = set(y_encoded)
        assert (
            len(unique_labels) == 2
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


# # Set paths

# In[4]:


# Directory with the training and testing datasets per plate (or combined per batch)
data_dir = pathlib.Path("data_splits")

# Directory with the trained models
model_dir = pathlib.Path("models")

# Directory with encoder
encoder_dir = pathlib.Path("encoder_results")

# Directory with the training indices
train_indices_dir = pathlib.Path("training_indices")

# Output directory the performance metrics
performance_metrics_dir = pathlib.Path("performance_metrics")
performance_metrics_dir.mkdir(exist_ok=True)

# Label being predicted
label = "Metadata_cell_type"


# ## Create dictionary with all relevant paths per plate to extract metrics

# In[5]:


# Get the list of encoder files
encoder_dir = pathlib.Path("./encoder_results")

# Extract model names from model filenames
model_names = set(
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
        "testing_data": pathlib.Path(
            data_dir / model / "testing_split.parquet"
        ).resolve(strict=True),
        "holdout_data": pathlib.Path(
            data_dir / model / "holdout_split.parquet"
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


# In[6]:


# For each model, print unique Metadata_heart_number per data split
for model_name, paths in models_dict.items():
    # load datasets
    downsample_train_df = pd.read_parquet(paths["training_data"])
    test_df = pd.read_parquet(paths["testing_data"])
    holdout_df = pd.read_parquet(paths["holdout_data"])

    # collect unique heart numbers per split and print
    print(f"Model: {model_name}")
    for split_name, df in [
        ("train", downsample_train_df),
        ("test", test_df),
        ("holdout", holdout_df),
    ]:
        if "Metadata_heart_number" not in df.columns:
            print(f"  {split_name}: Metadata_heart_number column not found")
            continue
        unique_hearts = pd.Series(df["Metadata_heart_number"].dropna().unique())
        # Print unique heart numbers horizontally, sorted and compact
        hearts = sorted(unique_hearts.tolist())
        if len(hearts) == 0:
            print(f"  {split_name} (0): None")
        else:
            hearts_str = ", ".join(str(h) for h in hearts)
            print(f"  {split_name} ({len(hearts)}): {hearts_str}")


# In[7]:


# Find the model key that corresponds to the "all hearts" model
model_key = next(
    (k for k in models_dict.keys() if "all" in k.lower() and "heart" in k.lower()),
    None,
)

if model_key is None:
    raise KeyError(
        f"No model key matching 'all' and 'heart' found. Available keys: {list(models_dict.keys())}"
    )

# Load holdout dataset for that model
holdout_path = models_dict[model_key]["holdout_data"]
holdout_df = pd.read_parquet(holdout_path)

# Filter to DMSO treatment only and drop rows missing required columns
dmso_holdout = holdout_df.loc[
    (holdout_df["Metadata_treatment"] == "DMSO")
    & holdout_df["Metadata_heart_number"].notna()
    & holdout_df["Metadata_Well"].notna()
].copy()

# Compute number of rows/cells per (heart, well)
cells_per_well = (
    dmso_holdout.groupby(["Metadata_heart_number", "Metadata_Well"])
    .size()
    .reset_index(name="n_cells")
)

# For each heart, collect sorted list of wells and corresponding counts
wells_per_heart = (
    cells_per_well.sort_values(["Metadata_heart_number", "Metadata_Well"])
    .groupby("Metadata_heart_number")
    .agg(
        heldout_wells=("Metadata_Well", lambda s: sorted(list(s))),
        cells_per_well_list=("n_cells", lambda s: list(s)),
        wells_with_counts=(
            "n_cells",
            lambda s, idx=None: None,
        ),  # placeholder column removed below
    )
    .reset_index()
)

# Replace wells_with_counts with a mapping from well -> count for clarity
wells_per_heart = wells_per_heart.drop(columns=["wells_with_counts"])
wells_per_heart["well_cell_counts"] = wells_per_heart["Metadata_heart_number"].map(
    lambda h: dict(
        cells_per_well.loc[
            cells_per_well["Metadata_heart_number"] == h, ["Metadata_Well", "n_cells"]
        ]
        .set_index("Metadata_Well")["n_cells"]
        .to_dict()
    )
)

# Print results
print(f"Model key: {model_key}")
print(
    "Held-out Metadata_Well values and cell counts per Metadata_heart_number for DMSO treatment:"
)
print(wells_per_heart.to_string(index=False))


# ## Extract metrics from the data splits applied to their respective models

# In[8]:


# Initialize results list
test_train_pr_results = []
test_train_probability_results = []

# Run through each model and get the PR results
for model_name, paths in models_dict.items():
    # Load the models and data
    final_model = load(paths["final_model"])
    shuffled_model = load(paths["shuffled_model"])
    label_encoder = load(paths["encoder_result"])
    downsample_train_df = pd.read_parquet(paths["training_data"])
    test_df = pd.read_parquet(paths["testing_data"])
    holdout_df = pd.read_parquet(paths["holdout_data"])

    print(f"Processing model: {model_name}")

    # Set dictionary with the data splits
    datasets = {"train": downsample_train_df, "test": test_df, "holdout": holdout_df}

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

            # Get PR curve results (global)
            pr_df = get_pr_curve_results(
                model=model, df=dataset, label=label, label_encoder=label_encoder
            )
            pr_df["model_type"] = model_type
            pr_df["dataset"] = dataset_name
            pr_df["model_name"] = model_name
            test_train_pr_results.append(pr_df)

            print(
                f"{model_name.upper()} | {model_name} | {model_type} | {dataset_name} → Done"
            )

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

# In[9]:


# Load in final and shuffled model for multi-class models
final_model = load(pathlib.Path(model_dir / "model_all_hearts_final_multiclass.joblib"))
shuffled_model = load(
    pathlib.Path(model_dir / "model_all_hearts_shuffled_multiclass.joblib")
)

# Set paths to data splits
training_data_path = pathlib.Path(
    data_dir
    / "model_all_hearts"
    / "training_split.parquet"  # Did not downsample for the multi-class model
).resolve(strict=True)
testing_data_path = pathlib.Path(
    data_dir / "model_all_hearts" / "testing_split.parquet"
).resolve(strict=True)
holdout_data_path = pathlib.Path(
    data_dir / "model_all_hearts" / "holdout_split.parquet"
).resolve(strict=True)


# In[10]:


# Initialize results list
multi_class_pr_results = []
heart_accuracy_results = []
cell_probs_results = []

# Load in label encoder for multi-class model
label_encoder = load(pathlib.Path(encoder_dir / "label_encoder_multi-class.joblib"))

# Set updated label variable
label = "Metadata_heart_failure_type"

for dataset_name, data_path in [
    ("train", training_data_path),
    ("test", testing_data_path),
    ("holdout", holdout_data_path),
]:
    # Load dataset
    df = pd.read_parquet(data_path)
    df[label] = df[label].fillna("Healthy")

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

            print(
                f"model_all_hearts_multiclass | {model_type} | {dataset_name} | {class_label} → Done"
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

