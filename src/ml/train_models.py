"""
AssureX Claim Engine — Step 3: 8-Model Training, Hyperparameter Tuning & Benchmark Report

Retrains and evaluates 8 machine learning classifiers on the 10,000-record dataset:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors (KNN)
8. Gaussian Naive Bayes

Features:
- 5-fold Stratified Cross-Validation with GridSearchCV hyperparameter optimization
- Evaluation on Validation Set (1,500 records) and Test Set (1,500 records) separately
- Per-class precision, recall, F1, and confusion matrix analysis for all 8 models
- Best model serialization to backend/model/best_model.joblib
- Comprehensive JSON and Markdown comparison reports in reports/
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    build_preprocessor,
    clean_dataframe,
    extract_feature_names,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CM_DIR = os.path.join(REPORTS_DIR, "confusion_matrices")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)


def load_datasets():
    """Load train (7,000), val (1,500), and test (1,500) CSV datasets."""
    train_path = os.path.join(DATA_DIR, "claims_train.csv")
    val_path = os.path.join(DATA_DIR, "claims_val.csv")
    test_path = os.path.join(DATA_DIR, "claims_test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Dataset splits not found! Run dataset_generator/generate_claims.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def get_tuning_grid():
    """
    Returns estimator dictionary and hyperparameter search grids for all 8 models.
    """
    models_and_grids = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "saga"],
            },
        ),
        "Decision_Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [6, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8, -1],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31, 63],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        ),
        "Naive_Bayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
            },
        ),
    }
    return models_and_grids


def plot_confusion_matrix(cm, model_name, save_path):
    """Render and save styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Likely Valid", "Likely Invalid", "Manual Review"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_evaluate_all():
    """Main training, cross-validation, tuning, and benchmark pipeline."""
    print("=== ASSUREX STEP 3: 8-MODEL HYPERPARAMETER TUNING & RETRAINING ===")

    train_df, val_df, test_df = load_datasets()
    print(f"Loaded splits: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

    # Clean dataframes
    X_train_raw = clean_dataframe(train_df)
    y_train = train_df["class_label"].map(LABEL_TO_INT).values

    X_val_raw = clean_dataframe(val_df)
    y_val = val_df["class_label"].map(LABEL_TO_INT).values

    X_test_raw = clean_dataframe(test_df)
    y_test = test_df["class_label"].map(LABEL_TO_INT).values

    # Preprocessing pipeline
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    feature_names = extract_feature_names(preprocessor)
    print(f"Engineered features vector dimension: {X_train_trans.shape[1]} features")

    # Save fitted preprocessor and label encoder
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump(LABEL_TO_INT, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print("Saved fitted preprocessor and label encoder to backend/model/")

    models_and_grids = get_tuning_grid()
    results = {}
    best_model_name = None
    best_model_obj = None
    best_test_f1 = -1.0

    cv_folder = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for m_key, (base_model, param_grid) in models_and_grids.items():
        pretty_name = m_key.replace("_", " ")
        print(f"\n--- Tuning & Training Model: {pretty_name} ---")

        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folder,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train_trans, y_train)
        train_time = time.time() - start_time

        best_estimator = grid_search.best_estimator_
        best_cv_f1 = grid_search.best_score_
        best_params = grid_search.best_params_
        print(f"  Best CV Weighted F1: {best_cv_f1*100:.2f}% | Params: {best_params}")

        # Benchmark Inference Time
        t_inf_start = time.time()
        _ = best_estimator.predict(X_val_trans)
        inf_time_ms = round(((time.time() - t_inf_start) / len(X_val_trans)) * 1000, 3)

        # Validation set evaluation
        y_val_pred = best_estimator.predict(X_val_trans)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

        # Per-class Validation metrics
        val_report = classification_report(y_val, y_val_pred, target_names=["Likely Valid", "Likely Invalid", "Manual Review Required"], output_dict=True)
        val_cm = confusion_matrix(y_val, y_val_pred).tolist()

        # Test set evaluation
        y_test_pred = best_estimator.predict(X_test_trans)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        # Per-class Test metrics
        test_report = classification_report(y_test, y_test_pred, target_names=["Likely Valid", "Likely Invalid", "Manual Review Required"], output_dict=True)
        test_cm = confusion_matrix(y_test, y_test_pred).tolist()

        # Save confusion matrix plot
        cm_plot_path = os.path.join(CM_DIR, f"cm_{m_key.lower()}.png")
        plot_confusion_matrix(np.array(test_cm), pretty_name, cm_plot_path)

        # Also save standalone model file
        model_save_path = os.path.join(MODEL_DIR, f"model_{m_key.lower()}.joblib")
        joblib.dump(best_estimator, model_save_path)

        print(f"  Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")
        print(f"  Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%")

        results[m_key] = {
            "model_name": pretty_name,
            "best_params": best_params,
            "cv_f1_mean": round(float(best_cv_f1), 4),
            "train_time_sec": round(train_time, 4),
            "inference_time_ms": inf_time_ms,
            "val_metrics": {
                "accuracy": round(float(val_acc), 4),
                "precision": round(float(val_prec), 4),
                "recall": round(float(val_rec), 4),
                "f1_weighted": round(float(val_f1), 4),
                "f1_likely_valid": round(float(val_report["Likely Valid"]["f1-score"]), 4),
                "f1_likely_invalid": round(float(val_report["Likely Invalid"]["f1-score"]), 4),
                "f1_manual_review": round(float(val_report["Manual Review Required"]["f1-score"]), 4),
                "confusion_matrix": val_cm,
            },
            "test_metrics": {
                "accuracy": round(float(test_acc), 4),
                "precision": round(float(test_prec), 4),
                "recall": round(float(test_rec), 4),
                "f1_weighted": round(float(test_f1), 4),
                "f1_likely_valid": round(float(test_report["Likely Valid"]["f1-score"]), 4),
                "f1_likely_invalid": round(float(test_report["Likely Invalid"]["f1-score"]), 4),
                "f1_manual_review": round(float(test_report["Manual Review Required"]["f1-score"]), 4),
                "confusion_matrix": test_cm,
            },
            "confusion_matrix_plot": f"reports/confusion_matrices/cm_{m_key.lower()}.png",
        }

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_name = pretty_name
            best_model_obj = best_estimator

    # Save best overall model artifact to best_model.joblib
    best_path = os.path.join(MODEL_DIR, "best_model.joblib")
    best_artifact = {
        "model": best_model_obj,
        "preprocessor": preprocessor,
        "model_name": best_model_name,
    }
    joblib.dump(best_artifact, best_path)
    print(f"\n=======================================================")
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"Test Weighted F1: {best_test_f1*100:.2f}%")
    print(f"Saved best model to: {best_path}")
    print(f"=======================================================")

    # Write JSON comparison report
    report_data = {
        "best_model": best_model_name,
        "dataset_info": {
            "total_records": 10000,
            "train_records": len(train_df),
            "val_records": len(val_df),
            "test_records": len(test_df),
        },
        "models": results,
    }

    json_report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Write Markdown comparison report
    md_report_path = os.path.join(REPORTS_DIR, "model_comparison.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# AssureX Claim Engine — 8-Model Training & Benchmark Comparison Report\n\n")
        f.write(f"**Dataset Scale:** 10,000 Unique Claims (7,000 Train / 1,500 Val / 1,500 Test)\n")
        f.write(f"**Winning Model:** **{best_model_name}** (Test Accuracy: **{results[best_model_name.replace(' ', '_')]['test_metrics']['accuracy']*100:.2f}%**)\n\n")
        f.write("## Performance Comparison Table (Test Set — 1,500 Records)\n\n")
        f.write("| Rank | Model Name | Test Accuracy | Weighted F1 | Precision | Recall | Manual Review F1 | 5-Fold CV F1 | Inference Time |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        # Sort models by test F1 descending
        sorted_models = sorted(results.values(), key=lambda x: x["test_metrics"]["f1_weighted"], reverse=True)
        for idx, m in enumerate(sorted_models, 1):
            tm = m["test_metrics"]
            f.write(f"| {idx} | **{m['model_name']}** | **{tm['accuracy']*100:.2f}%** | {tm['f1_weighted']:.4f} | {tm['precision']:.4f} | {tm['recall']:.4f} | {tm['f1_manual_review']:.4f} | {m['cv_f1_mean']:.4f} | {m['inference_time_ms']} ms |\n")

        f.write("\n## Hyperparameter Tuning Summary\n\n")
        for m in sorted_models:
            f.write(f"- **{m['model_name']}**: Best Params = `{json.dumps(m['best_params'])}` | CV F1 = {m['cv_f1_mean']:.4f}\n")

        f.write("\n## Per-Class Metric Breakdown (Test Set)\n\n")
        f.write("| Model | Likely Valid F1 | Likely Invalid F1 | Manual Review F1 | Test Accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m in sorted_models:
            tm = m["test_metrics"]
            f.write(f"| **{m['model_name']}** | {tm['f1_likely_valid']:.4f} | {tm['f1_likely_invalid']:.4f} | **{tm['f1_manual_review']:.4f}** | {tm['accuracy']*100:.2f}% |\n")

    print(f"Saved JSON Report: {json_report_path}")
    print(f"Saved Markdown Report: {md_report_path}")


if __name__ == "__main__":
    train_and_evaluate_all()
