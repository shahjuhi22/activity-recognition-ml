"""
train_model.py
--------------
Loads the UCI HAR dataset, trains Logistic Regression and Random Forest
classifiers, evaluates them, and saves the best model to model.pkl.

Run this after downloading the dataset:
    python train_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "UCI HAR Dataset")
OUTPUT_DIR = "outputs"        # Where plots and the model are saved
MODEL_PATH = "model.pkl"

ACTIVITY_LABELS = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split_name):
    """Load X features, y labels for a given split ('train' or 'test')."""
    split_dir = os.path.join(DATA_DIR, split_name)

    X = pd.read_csv(
        os.path.join(split_dir, f"X_{split_name}.txt"),
        sep=r"\s+", header=None
    )
    y = pd.read_csv(
        os.path.join(split_dir, f"y_{split_name}.txt"),
        sep=r"\s+", header=None
    ).squeeze()

    return X, y


def load_feature_names():
    """Read the 561 feature names from features.txt."""
    features_path = os.path.join(DATA_DIR, "features.txt")
    features = pd.read_csv(features_path, sep=r"\s+", header=None)
    names = features[1].tolist()

    # Make names unique (some are duplicated in the dataset)
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    return unique_names


def load_data():
    """Load train/test splits with named columns."""
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)

    feature_names = load_feature_names()

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    X_train.columns = feature_names
    X_test.columns = feature_names

    # Map integer labels → human-readable names
    y_train = y_train.map(ACTIVITY_LABELS)
    y_test = y_test.map(ACTIVITY_LABELS)

    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  Activities       : {list(ACTIVITY_LABELS.values())}")

    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def explore_data(X_train, X_test, y_train, y_test):
    """Print basic EDA stats and check for missing values."""
    print("\n" + "=" * 60)
    print("STEP 2: Data Exploration")
    print("=" * 60)

    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    print(f"  Missing values in training set : {missing_train}")
    print(f"  Missing values in test set     : {missing_test}")

    print("\n  Activity distribution (train):")
    dist = y_train.value_counts()
    for activity, count in dist.items():
        print(f"    {activity:<25} {count} samples")

    # Save activity distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training distribution
    dist.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Activity Distribution — Training Set", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Activity")
    axes[0].set_ylabel("Number of Samples")
    axes[0].tick_params(axis="x", rotation=30)

    # Test distribution
    y_test.value_counts().plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].set_title("Activity Distribution — Test Set", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Activity")
    axes[1].set_ylabel("Number of Samples")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activity_distribution.png"), dpi=150)
    plt.close()
    print("\n  Saved: outputs/activity_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with a reasonable max_iter."""
    print("\n  Training Logistic Regression ...", end=" ", flush=True)
    model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs", C=1.0)
    model.fit(X_train, y_train)
    print("Done.")
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("  Training Random Forest       ...", end=" ", flush=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   n_jobs=-1, max_depth=None)
    model.fit(X_train, y_train)
    print("Done.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    """Return a dict of evaluation metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    return {
        "name": model_name,
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_comparison(results):
    """Print a neat comparison table of both models."""
    print("\n" + "=" * 60)
    print("STEP 4: Model Comparison")
    print("=" * 60)
    print(f"  {'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r['name']:<28} {r['accuracy']:>8.4f}  "
            f"{r['precision']:>9.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}"
        )


def plot_comparison(results):
    """Bar chart comparing both models on all four metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = [m.capitalize() for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, [results[0][m] for m in metrics],
                   width, label=results[0]["name"], color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width / 2, [results[1][m] for m in metrics],
                   width, label=results[1]["name"], color="coral", edgecolor="black")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels on bars
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: outputs/model_comparison.png")


def plot_confusion_matrix(y_test, y_pred, model_name, activity_labels):
    """Plot and save the confusion matrix for the best model."""
    cm = confusion_matrix(y_test, y_pred, labels=activity_labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activity_labels, yticklabels=activity_labels,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  Saved: outputs/confusion_matrix.png")


def save_metrics(results, best_result):
    """Persist metrics to a CSV so the Streamlit app can reload them."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1 Score": round(r["f1"], 4),
            "Best": r["name"] == best_result["name"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    print("  Saved: outputs/metrics.csv")


def save_classification_report(y_test, y_pred, model_name):
    """Save the classification report to a text file."""
    report = classification_report(y_test, y_pred)
    path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(path, "w") as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  Saved: outputs/classification_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2. Explore
    explore_data(X_train, X_test, y_train, y_test)

    # 3. Train models
    print("\n" + "=" * 60)
    print("STEP 3: Training Models")
    print("=" * 60)
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 4. Evaluate
    lr_result = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results = [lr_result, rf_result]

    print_comparison(results)

    # 5. Select best model (by accuracy)
    best_result = max(results, key=lambda r: r["accuracy"])
    print(f"\n  [*] Best model: {best_result['name']} "
          f"(Accuracy = {best_result['accuracy']:.4f})")

    # 6. Plots
    print("\n" + "=" * 60)
    print("STEP 5: Saving Plots")
    print("=" * 60)
    plot_comparison(results)

    activity_labels = list(ACTIVITY_LABELS.values())
    plot_confusion_matrix(y_test, best_result["y_pred"],
                          best_result["name"], activity_labels)

    # 7. Classification report
    save_classification_report(y_test, best_result["y_pred"], best_result["name"])

    # 8. Metrics CSV
    save_metrics(results, best_result)

    # 9. Save best model + feature names
    print("\n" + "=" * 60)
    print("STEP 6: Saving Model")
    print("=" * 60)
    joblib.dump(
        {"model": best_result["model"], "feature_names": feature_names},
        MODEL_PATH,
    )
    print(f"  Model saved to: {MODEL_PATH}")

    # 10. Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model : {best_result['name']}")
    print(f"  Accuracy   : {best_result['accuracy'] * 100:.2f}%")
    print(f"  F1 Score   : {best_result['f1'] * 100:.2f}%")
    print("\n  Run the web app next:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
