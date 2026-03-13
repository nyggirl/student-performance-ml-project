from __future__ import annotations

"""
Starter file for the next phase of the project.

This file is intentionally only a scaffold so that the codebase stays aligned with the
first progress report: preprocessing and exploratory analysis are done, while model
comparison is the immediate next step.
"""

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Starter script for the model-training phase.")
    parser.add_argument("--input", required=True, help="Path to processed training CSV.")
    parser.add_argument("--target", default="performance_label", help="Target column name.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.input}")

    print("Model training stage is the next milestone.")
    print(f"Loaded dataset with shape: {df.shape}")
    print("Planned models:")
    print("- Logistic Regression (baseline)")
    print("- Decision Tree")
    print("- Random Forest")
    print("Next: connect this script to the preprocessing pipeline and evaluation metrics.")


if __name__ == "__main__":
    main()
