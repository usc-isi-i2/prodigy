from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def classification_metrics(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict:
    if len(classes) == 2:
        auc = roc_auc_score(y, probability[:, 1], labels=classes)
    else:
        binary = label_binarize(y, classes=classes)
        auc = roc_auc_score(binary, probability, average="macro", multi_class="ovr")
    return {
        "roc_auc_ovr_macro": float(auc),
        "f1_macro": float(f1_score(y, prediction, average="macro")),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }
