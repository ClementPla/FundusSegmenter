from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet


from sklearn.metrics import mean_squared_error
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline


def binary_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def multiclass_macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.sum(axis=1) - np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity_per_class = tn / (tn + fp)
    return np.mean(specificity_per_class)


def binary_metric(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": binary_specificity(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def multiclass_metric(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_eval_linear_regression(
    df_train: pd.DataFrame,
    df_test: dict[str, pd.DataFrame],
    columns_to_drop: list[str] = None,
):
    X_train = df_train.drop(columns=["label"] + (columns_to_drop or []))
    y_train = df_train["label"]
    print(
        f"Found {len(X_train)} training samples, with N {len(y_train.unique())} unique labels."
    )
    model = make_pipeline(StandardScaler(), ElasticNet(l1_ratio=0.05))
    model.fit(X_train, y_train)

    results = {}
    for name, df in df_test.items():
        X_test = df.drop(columns=["label"] + (columns_to_drop or []))
        y_test = df["label"]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {"mse": mse}
        # Clamp results to valid range [0, 5    ] for DR grading
        y_pred_clamped = np.floor(np.clip(y_pred, 0, 5))
        results[name].update(multiclass_metric(y_test, y_pred_clamped))

    # Format as a DataFrame for better display: columns are test set, rows are metrics
    results_df = pd.DataFrame(results)
    return results_df, model


def train_and_eval_logistic_regression(
    df_train: pd.DataFrame,
    df_test: dict[str, pd.DataFrame],
    columns_to_drop: list[str] = None,
    is_binary: bool = True,
):
    X_train = df_train.drop(columns=["label"] + (columns_to_drop or []))
    y_train = df_train["label"]
    if is_binary:
        y_train = y_train > 1  # Convert to binary: 0-1 vs 2-4
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    results = {}
    for name, df in df_test.items():
        X_test = df.drop(columns=["label"] + (columns_to_drop or []))
        y_test = df["label"]
        if is_binary and len(y_test.unique()) > 2:
            y_test = y_test > 1  # Convert to binary: 0-1 vs 2-4
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        if is_binary:
            results[name] = binary_metric(y_test, y_pred, y_proba)
        else:
            results[name] = multiclass_metric(y_test, y_pred)

    # Format as a DataFrame for better display: columns are test set, rows are metrics
    results_df = pd.DataFrame(results)
    return results_df
