import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from config import DEFAULT_THRESHOLD

def evaluate_model_performance(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = "Model",
    plot_cm: bool = False
) -> Dict[str, float]:
    """
    Evaluates binary classification metrics and optionally plots Confusion Matrix.
    Features defensive type, range, shape, and metric computation validations.
    """
    # 1. Type and Threshold Validations
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        raise ValueError(f"[Error - Metrics]: Threshold must be a float between 0.0 and 1.0, got {threshold}")

    try:
        y_true_arr = np.asarray(y_true).ravel()
        y_probs_arr = np.asarray(y_probs).ravel()
    except Exception as e:
        raise TypeError(f"[Error - Metrics]: Failed to convert inputs to 1D numpy arrays: {e}")

    # 2. Shape and Content Checks
    if y_true_arr.shape[0] != y_probs_arr.shape[0]:
        raise ValueError(
            f"[Error - Metrics]: Length mismatch. y_true has length {y_true_arr.shape[0]}, "
            f"but y_probs has length {y_probs_arr.shape[0]}"
        )

    if y_true_arr.size == 0:
        raise ValueError("[Error - Metrics]: Input arrays are empty.")

    if np.isnan(y_probs_arr).any() or np.isnan(y_true_arr).any():
        raise ValueError("[Error - Metrics]: Inputs contain NaN values.")

    unique_classes = np.unique(y_true_arr)
    if not np.all(np.isin(unique_classes, [0, 1])):
        raise ValueError(f"[Error - Metrics]: y_true must contain only binary labels (0 and 1), found {unique_classes}")

    try:
        # 3. Decision Threshold Application
        y_pred = (y_probs_arr >= threshold).astype(int)

        # 4. Mandatory Sprint Metrics Calculation
        roc_auc = roc_auc_score(y_true_arr, y_probs_arr)
        pr_auc = average_precision_score(y_true_arr, y_probs_arr)
        precision = precision_score(y_true_arr, y_pred, zero_division=0)
        recall = recall_score(y_true_arr, y_pred, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)
        cm = confusion_matrix(y_true_arr, y_pred)

        metrics = {
            "Threshold": float(threshold),
            "ROC-AUC": round(float(roc_auc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "F1": round(float(f1), 4)
        }

        # 5. Confusion Matrix Visualization (Sadece plot_cm=True verilirse açılır)
        if plot_cm:
            try:
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Retained (0)', 'Churned (1)'],
                            yticklabels=['Retained (0)', 'Churned (1)'])
                plt.title(f"{model_name} (Threshold: {threshold})")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.tight_layout()
                plt.show()
            except Exception as plot_err:
                print(f"[Warning - Metrics Plot]: Failed to render Confusion Matrix: {plot_err}")

        return metrics

    except Exception as e:
        print(f"[Error - Metrics]: Metric computation failed: {e}")
        raise