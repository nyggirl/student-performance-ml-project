from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Logistic Regression model.")
    parser.add_argument("--train", required=True, help="Path to training CSV.")
    parser.add_argument("--test", required=True, help="Path to testing CSV.")
    parser.add_argument("--target", default="performance_label", help="Target column name.")
    parser.add_argument("--output", default="outputs", help="Directory to save results.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Split features and target
    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    # IMPORTANT: convert all features to numeric (simple approach)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align columns (in case one-hot encoding differs)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save results
    with open(output_dir / "model_results.txt", "w") as f:
        f.write("Logistic Regression Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("Model training complete.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Results saved to: {output_dir / 'model_results.txt'}")


if __name__ == "__main__":
    main()