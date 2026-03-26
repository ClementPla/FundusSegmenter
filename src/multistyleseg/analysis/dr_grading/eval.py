import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def evaluate_mutual_information(df):
    """
    Measure how much information lesion bins share with diagnosis.
    """

    # Bin both variables
    bins = [-np.inf, 5, 10, 20, np.inf]
    df["N_Sig_binned"] = pd.cut(df["N Significant"], bins=bins, labels=[0, 1, 2, 3])

    # Drop rows where binning failed
    df = df.dropna(subset=["N_Sig_binned"])
    df["N_Sig_binned"] = df["N_Sig_binned"].astype(int)

    diag_encoded = LabelEncoder().fit_transform(df["Diagnosis"].astype(str))

    # Mutual Information
    mi = mutual_info_score(diag_encoded, df["N_Sig_binned"])

    # Normalized Mutual Information (0-1 scale)
    nmi = normalized_mutual_info_score(diag_encoded, df["N_Sig_binned"])

    return {
        "Mutual Information": mi,
        "Normalized MI": nmi,
    }


def evaluate_referable_detection(df, lesion_cols):
    """
    Evaluate ability to detect referable (sight-threatening) DR.
    This is the most clinically relevant binary task.
    """

    df_binary = df.dropna(subset=["Referable"]).copy()
    X = df_binary[lesion_cols].values.reshape(-1, 1)
    y = df_binary["Referable"].values.astype(int)

    results = {}

    # Simple threshold on N Significant
    for threshold in [1, 5, 10, 15, 20]:
        y_pred = (df_binary[lesion_cols] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        results[f"Threshold ≥{threshold}"] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
            "Accuracy": (tp + tn) / len(y),
        }

    # ROC-AUC for N Significant alone
    auc_n_sig = roc_auc_score(y, df_binary[lesion_cols])
    results[f"{lesion_cols} (AUC)"] = {"AUC-ROC": auc_n_sig}

    # ROC-AUC for all lesion features (logistic regression)
    if len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        cv_auc = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")
        results["All lesions (LR, 5-fold CV)"] = {
            "AUC-ROC mean": cv_auc.mean(),
            "AUC-ROC std": cv_auc.std(),
        }

    return results


def evaluate_ordinal_classification(df, lesion_column="N Significant"):
    """
    Evaluate lesion-based binning against diagnosis using ordinal metrics.
    Quadratic Weighted Kappa is the gold standard for ordinal medical grading.
    """

    # Bin the specified lesion column into ordinal groups
    bins = [-np.inf, 5, 10, 20, np.inf]
    labels = [0, 1, 2, 3]  # Numeric for kappa calculation
    df["N_Sig_binned"] = pd.cut(df[lesion_column], bins=bins, labels=labels)

    # Bin Diagnosis severity into comparable groups
    diag_bins = [-np.inf, 4, 7, 10, np.inf]  # No DR, Mild, Moderate, Severe+
    df["Diag_binned"] = pd.cut(
        df["Diagnosis_severity"], bins=diag_bins, labels=labels, right=False
    )

    # Drop rows where binning failed
    df = df.dropna(subset=["N_Sig_binned", "Diag_binned"])
    df["N_Sig_binned"] = df["N_Sig_binned"].astype(int)
    df["Diag_binned"] = df["Diag_binned"].astype(int)

    # Cohen's Kappa (unweighted)
    kappa = cohen_kappa_score(df["Diag_binned"], df["N_Sig_binned"])

    # Quadratic Weighted Kappa (penalizes larger disagreements more)
    qwk = cohen_kappa_score(df["Diag_binned"], df["N_Sig_binned"], weights="quadratic")

    # Linear Weighted Kappa
    lwk = cohen_kappa_score(df["Diag_binned"], df["N_Sig_binned"], weights="linear")

    return {
        "Cohen Kappa (unweighted)": kappa,
        "Linear Weighted Kappa": lwk,
        "Quadratic Weighted Kappa": qwk,
    }


def evaluate_correlations(df, lesion_cols, target_col="Diagnosis_severity"):
    """Compute correlation between lesion counts and diagnosis severity."""

    results = {}

    for col in lesion_cols:
        # Pearson (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(df[col], df[target_col])

        # Spearman (monotonic relationship - better for ordinal)
        spearman_r, spearman_p = stats.spearmanr(df[col], df[target_col])

        # Kendall's tau (rank correlation, robust)
        kendall_tau, kendall_p = stats.kendalltau(df[col], df[target_col])

        results[col] = {
            "Pearson r": pearson_r,
            "Pearson p": pearson_p,
            "Spearman ρ": spearman_r,
            "Spearman p": spearman_p,
            "Kendall τ": kendall_tau,
            "Kendall p": kendall_p,
        }

    return pd.DataFrame(results).T


# =============================================================================
# 6. PER-LESION-TYPE ANALYSIS
# =============================================================================


def evaluate_individual_lesions(df, lesion_cols, target="Referable"):
    """
    Evaluate which lesion types are most predictive.
    """

    df_analysis = df.dropna(subset=[target]).copy()
    y = df_analysis[target].values.astype(int)

    results = {}
    for col in lesion_cols:
        if col == "N Significant":
            continue

        x = df_analysis[col].values

        # AUC-ROC
        try:
            auc = roc_auc_score(y, x)
        except:
            auc = np.nan

        # Spearman correlation with severity
        spearman_r, _ = stats.spearmanr(
            df_analysis[col], df_analysis["Diagnosis_severity"]
        )

        results[col] = {
            "AUC (Referable)": auc,
            "Spearman (Severity)": spearman_r,
        }

    return pd.DataFrame(results).T.sort_values("AUC (Referable)", ascending=False)


def create_summary_scorecard(df, lesion_cols):
    """Create a summary of all evaluation metrics."""

    scorecard = {
        "Metric": [],
        "Value": [],
        "Interpretation": [],
    }

    # Correlation
    spearman_r, _ = stats.spearmanr(df["N Significant"], df["Diagnosis_severity"])
    scorecard["Metric"].append("Spearman ρ (N Sig vs Severity)")
    scorecard["Value"].append(f"{spearman_r:.3f}")
    if spearman_r > 0.7:
        interp = "Strong monotonic relationship"
    elif spearman_r > 0.5:
        interp = "Moderate relationship"
    else:
        interp = "Weak relationship"
    scorecard["Interpretation"].append(interp)

    # QWK
    df_temp = df.copy()
    bins = [-np.inf, 5, 10, 20, np.inf]
    df_temp["N_Sig_binned"] = pd.cut(
        df_temp["N Significant"], bins=bins, labels=[0, 1, 2, 3]
    )
    diag_bins = [-np.inf, 4, 7, 10, np.inf]
    df_temp["Diag_binned"] = pd.cut(
        df_temp["Diagnosis_severity"], bins=diag_bins, labels=[0, 1, 2, 3], right=False
    )
    df_temp = df_temp.dropna(subset=["N_Sig_binned", "Diag_binned"])
    df_temp["N_Sig_binned"] = df_temp["N_Sig_binned"].astype(int)
    df_temp["Diag_binned"] = df_temp["Diag_binned"].astype(int)
    qwk = cohen_kappa_score(
        df_temp["Diag_binned"], df_temp["N_Sig_binned"], weights="quadratic"
    )
    scorecard["Metric"].append("Quadratic Weighted Kappa")
    scorecard["Value"].append(f"{qwk:.3f}")
    if qwk > 0.8:
        interp = "Almost perfect agreement"
    elif qwk > 0.6:
        interp = "Substantial agreement"
    elif qwk > 0.4:
        interp = "Moderate agreement"
    else:
        interp = "Fair/Poor agreement"
    scorecard["Interpretation"].append(interp)

    # AUC for referable
    df_ref = df.dropna(subset=["Referable"])
    auc = roc_auc_score(df_ref["Referable"], df_ref["N Significant"])
    scorecard["Metric"].append("AUC-ROC (Referable DR)")
    scorecard["Value"].append(f"{auc:.3f}")
    if auc > 0.9:
        interp = "Excellent discrimination"
    elif auc > 0.8:
        interp = "Good discrimination"
    elif auc > 0.7:
        interp = "Fair discrimination"
    else:
        interp = "Poor discrimination"
    scorecard["Interpretation"].append(interp)

    # NMI
    df_nmi = df.copy()
    df_nmi["N_Sig_binned"] = pd.cut(
        df_nmi["N Significant"], bins=bins, labels=[0, 1, 2, 3]
    )
    df_nmi = df_nmi.dropna(subset=["N_Sig_binned"])
    df_nmi["N_Sig_binned"] = df_nmi["N_Sig_binned"].astype(int)
    nmi = normalized_mutual_info_score(
        LabelEncoder().fit_transform(df_nmi["Diagnosis"].astype(str)),
        df_nmi["N_Sig_binned"],
    )
    scorecard["Metric"].append("Normalized Mutual Information")
    scorecard["Value"].append(f"{nmi:.3f}")
    if nmi > 0.5:
        interp = "High shared information"
    elif nmi > 0.3:
        interp = "Moderate shared information"
    else:
        interp = "Low shared information"
    scorecard["Interpretation"].append(interp)

    return pd.DataFrame(scorecard)
