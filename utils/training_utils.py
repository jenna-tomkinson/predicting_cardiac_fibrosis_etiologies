"""
This utility file contains the function to load in training data as final or shuffled to be used in training a machine learning model.
"""

import pathlib

import numpy as np
import pandas as pd

# set numpy seed to make random operations (shuffling data) reproducible
np.random.seed(0)


def get_X_y_data(
    df: pd.DataFrame, label: str, shuffle_features: bool = False
) -> tuple[pd.DataFrame, np.array]:
    """Get X (feature space) and labels (predicting class) from pandas Data frame

    Args:
        df (pd.DataFrame): Data frame containing morphology.
        label (str): Name of the Metadata column being used as the predicting class
        shuffle_features (bool, optional): Shuffle the feature columns to get a shuffled dataset. 
            Defaults to False.

    Returns:
        Tuple[pd.DataFrame, np.array]: Returns  dataframe of the feature space (X) and np.array for 
            the predicting class (y)
    """
    # Remove "Metadata" columns from df, leaving only the feature space as a dataframe
    feature_columns = df.columns[~df.columns.str.contains("Metadata")]
    X = df[feature_columns]

    # Extract class label
    y = df.loc[:, [label]].values
    # Make labels as array for use in machine learning
    y = np.ravel(y)

    # If shuffle is True, shuffle the rows within each column independently for the feature space
    if shuffle_features:
        X = X.copy()  # Avoid in-place modification of the original data
        for column in X.columns:
            X[column] = np.random.permutation(X[column].values)

    return X, y


def load_data(
    path_to_data: pathlib.Path, label: str, shuffle_features: bool = False
) -> tuple[np.array, np.array]:
    """Load in data from a path as X (feature space) and labels (predicting class)

    Args:
        path_to_data (pathlib.Path): Path to the CSV contain morphology data that you want to load in. Expected format is CSV file.
        label (str): Name of the Metadata column being used as the predicting class
        shuffle_features (bool, optional): Shuffle the feature columns to get a shuffled dataset. Defaults to False.

    Returns:
        Tuple[np.array, np.array]: Returns np.arrays for the feature space (X) and the predicting class (y)
    """
    # Load in data frame from CSV, if not CSV file then return error
    if path_to_data.suffix.lower() == ".csv":
        # Load the CSV file
        df = pd.read_csv(path_to_data, index_col=0)
    else:
        print("File does not have a CSV extension. Current expected input is CSV.")

    # Get X, y data from loaded in data frame
    X, y = get_X_y_data(df=df, label=label, shuffle_features=shuffle_features)

    return X, y


def downsample_data(data: [pathlib.Path | pd.DataFrame], label: str) -> pd.DataFrame:
    """Load in data from a path or use an existing DataFrame and downsample to the lowest class,
    returning a DataFrame to use for retrieving X, y data.

    Args:
        data (pathlib.Path | pd.DataFrame): Path to CSV file or an already loaded DataFrame.
        label (str): Name of the Metadata column being used as the predicting class.

    Returns:
        pd.DataFrame: Downsampled DataFrame on success.

    Raises:
        TypeError: If input is neither a pathlib.Path nor a pandas.DataFrame.
        ValueError: If provided path does not point to a CSV file.
    """
    # If data is a Path, load the CSV file
    if isinstance(data, pathlib.Path):
        if data.suffix.lower() == ".csv":
            df = pd.read_csv(data, index_col=0)
        else:
            raise ValueError(
                "File does not have a CSV extension. Expected input is CSV."
            )
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a Path or a pandas DataFrame.")

    # Find class with lowest sample from label
    min_samples = df[label].value_counts().min()

    # Downsample classes to the lowest label
    df_downsampled = df.groupby(label, group_keys=False).apply(
        lambda x: x.sample(min_samples)
    )

    return df_downsampled


def get_heart_grouped_folds(
    df: pd.DataFrame,
    group_col: str,
    strata_col: str,
    n_splits: int = 3,
    random_state: int = 0,
) -> list[tuple[np.array, np.array]]:
    """Get cross validation folds where all single cells from a heart are in the same fold and
    the hearts from each class (e.g., heart failure subtype) are spread evenly across the folds.

    Args:
        df (pd.DataFrame): Data frame containing the training data (in the same row order as X and y).
        group_col (str): Name of the Metadata column with the heart each single cell comes from.
        strata_col (str): Name of the Metadata column with the class of each heart to spread across
            folds (NaNs are treated as their own class).
        n_splits (int, optional): Number of folds. Defaults to 3.
        random_state (int, optional): Seed for randomly ordering the hearts. Defaults to 0.

    Returns:
        list[tuple[np.array, np.array]]: List of (train, validation) positional indices per fold,
            which can be used as the `cv` parameter in scikit-learn.

    Raises:
        ValueError: If a class has fewer hearts than the number of folds.
    """
    rng = np.random.default_rng(random_state)

    # Get the class per heart
    hearts_df = df[[group_col, strata_col]].drop_duplicates()
    hearts_df[strata_col] = hearts_df[strata_col].fillna("NaN")

    # Assign the hearts from each class across the folds (one at a time in random order)
    heart_to_fold = {}
    for strata, strata_df in hearts_df.groupby(strata_col):
        hearts = rng.permutation(np.sort(strata_df[group_col].unique()))
        if len(hearts) < n_splits:
            raise ValueError(
                f"Class '{strata}' has {len(hearts)} hearts, which is fewer than {n_splits} folds."
            )
        for i, heart in enumerate(hearts):
            heart_to_fold[heart] = i % n_splits

    # Get the positional indices for the training and validation data per fold
    fold_per_row = df[group_col].map(heart_to_fold).values
    return [
        (np.where(fold_per_row != fold)[0], np.where(fold_per_row == fold)[0])
        for fold in range(n_splits)
    ]
