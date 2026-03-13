from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import create_binary_target, load_dataset, split_feature_types


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess student performance data for later ML training.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", default="outputs", help="Directory to save processed outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    df = create_binary_target(df)

    numeric_cols, categorical_cols = split_feature_types(df)

    # Keep the original cleaned tabular file for easy inspection and reproducibility.
    df.to_csv(output_dir / "processed_student_data.csv", index=False)

    target_col = "performance_label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    metadata = {
        "row_count": len(df),
        "column_count": df.shape[1],
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_column": target_col,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save split files so model training can be done in the next stage.
    X_train.assign(**{target_col: y_train.values}).to_csv(output_dir / "train_split.csv", index=False)
    X_test.assign(**{target_col: y_test.values}).to_csv(output_dir / "test_split.csv", index=False)

    # Fit-transform here only to verify that preprocessing runs successfully.
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    with open(output_dir / "preprocessing_check.txt", "w", encoding="utf-8") as f:
        f.write("Preprocessing pipeline executed successfully.\n")
        f.write(f"Transformed training shape: {transformed_train.shape}\n")
        f.write(f"Transformed testing shape: {transformed_test.shape}\n")

    print("Preprocessing complete.")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
