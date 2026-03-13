from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


POSSIBLE_CATEGORICAL_TARGETS = [
    "performance_label",
    "target",
    "label",
    "passed",
    "final_result",
]


COMMON_NON_FEATURE_COLUMNS = {
    "performance_label",
    "target",
    "label",
    "passed",
    "final_result",
}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. Please place your CSV in the data folder or update the path."
        )
    return pd.read_csv(path, sep=";")



def create_binary_target(df: pd.DataFrame, pass_mark: int = 10) -> pd.DataFrame:
    """
    Create a binary classification target if the dataset does not already include one.

    Common student performance datasets include G3 as the final grade.
    Here we convert it into a binary outcome:
    - 1 = pass / satisfactory performance
    - 0 = below threshold
    """
    df = df.copy()

    if any(col in df.columns for col in POSSIBLE_CATEGORICAL_TARGETS):
        return df

    if "G3" not in df.columns:
        raise ValueError(
            "No target column found. Expected an existing label column or a 'G3' column to build one."
        )

    df["performance_label"] = (df["G3"] >= pass_mark).astype(int)
    return df



def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical feature columns."""
    feature_df = df.drop(columns=[c for c in COMMON_NON_FEATURE_COLUMNS if c in df.columns], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols
