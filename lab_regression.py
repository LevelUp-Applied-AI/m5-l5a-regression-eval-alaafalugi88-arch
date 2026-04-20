"""
Module 5 Week A — Lab: Regression & Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with optional stratification."""

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # إذا target تصنيف → stratify
    if y.nunique() <= 10:  # heuristic بسيطة
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )


def build_logistic_pipeline():
    """Build Logistic Regression Pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipeline


def build_ridge_pipeline():
    """Build Ridge Regression Pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate classifier."""

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate regressor."""

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": mae,
        "r2": r2
    }


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation."""

    try:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="accuracy"
        )
        return scores
    except Exception as e:
        print("CV Error:", e)
        return None


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # =========================
        # Classification
        # =========================
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                            "num_support_calls", "senior_citizen",
                            "has_partner", "has_dependents"]

        df_cls = df[numeric_features + ["churned"]].dropna()

        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split

            pipe = build_logistic_pipeline()
            metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)

            print("\nLogistic Regression Metrics:")
            print(metrics)

            scores = run_cross_validation(pipe, X_train, y_train)
            if scores is not None:
                print(f"\nCV Scores: {scores}")
                print(f"Mean: {scores.mean():.3f} +/- {scores.std():.3f}")

        # =========================
        # Regression
        # =========================
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                      "senior_citizen", "has_partner", "has_dependents",
                      "monthly_charges"]].dropna()

        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg

            ridge_pipe = build_ridge_pipeline()
            reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)

            print("\nRidge Regression Metrics:")
            print(reg_metrics)