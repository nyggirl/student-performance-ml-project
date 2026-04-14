from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare multiple models.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", default="performance_label")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    # Convert categorical → numeric
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    print("Model Comparison Results:\n")

    for name, model in models.items():
        acc = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        results[name] = acc
        print(f"{name}: {acc:.4f}")

    # Save comparison results
    with open(output_dir / "model_comparison.txt", "w") as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 40 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")

    print(f"\nResults saved to: {output_dir / 'model_comparison.txt'}")

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Save top 10 important features
    importance_df.head(10).to_csv(output_dir / "feature_importance.csv", index=False)

    print("\nTop 10 Important Features:")
    print(importance_df.head(10))


if __name__ == "__main__":
    main()