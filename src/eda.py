from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import create_binary_target, load_dataset, split_feature_types


def save_missing_value_report(df: pd.DataFrame, output_dir: Path) -> None:
    report = df.isnull().sum().sort_values(ascending=False)
    report.to_csv(output_dir / "missing_value_report.csv", header=["missing_count"])



def save_basic_summary(df: pd.DataFrame, output_dir: Path) -> None:
    with open(output_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("Student Performance Dataset Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"- {col} ({df[col].dtype})\n")

        f.write("\nNumeric Summary:\n")
        f.write(df.describe(include="number").to_string())
        f.write("\n\nCategorical Summary:\n")
        categorical_summary = df.describe(include="object")
        if len(categorical_summary.columns) > 0:
            f.write(categorical_summary.to_string())
        else:
            f.write("No categorical columns detected.")



def save_target_plot(df: pd.DataFrame, output_dir: Path) -> None:
    if "performance_label" not in df.columns:
        return

    counts = df["performance_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Performance Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png")
    plt.close()



def save_numeric_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols, _ = split_feature_types(df)

    selected = [col for col in ["studytime", "failures", "absences", "G1", "G2", "G3"] if col in numeric_cols]
    if not selected:
        selected = numeric_cols[:4]

    for col in selected:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png")
        plt.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="Run early exploratory analysis for student performance data.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", default="outputs", help="Directory for reports and plots.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    df = create_binary_target(df)

    save_basic_summary(df, output_dir)
    save_missing_value_report(df, output_dir)
    save_target_plot(df, output_dir)
    save_numeric_histograms(df, output_dir)

    print("EDA complete.")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
