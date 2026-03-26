
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def plot_roc_curves(df, lesion_cols):
    """Plot ROC curves for referable DR detection."""
    
    df_analysis = df.dropna(subset=['Referable']).copy()
    y = df_analysis['Referable'].values.astype(int)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ROC for each lesion type
    for col in lesion_cols:
        fpr, tpr, _ = roc_curve(y, df_analysis[col])
        auc = roc_auc_score(y, df_analysis[col])
        ax.plot(fpr, tpr, label=f'{col} (AUC={auc:.3f})')
    
    # ROC for combined model (logistic regression)
    X = df_analysis[lesion_cols].values
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    y_prob = lr.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    ax.plot(fpr, tpr, 'k--', linewidth=2, label=f'Combined LR (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', label='Random')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves for Referable DR Detection')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

