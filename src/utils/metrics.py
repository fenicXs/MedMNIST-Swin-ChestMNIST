from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class Metrics:
    auc: float
    acc: float
    per_label_auc: Dict[int, float]


def multilabel_auc_and_acc(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Metrics:
    """Compute mean ROC-AUC (per label) and micro accuracy for multi-label classification.

    Args:
        y_true: (N, L) binary labels in {0,1}
        y_score: (N, L) predicted probabilities in [0,1]
        threshold: threshold for ACC

    Returns:
        Metrics with:
          - auc: mean over labels (skipping undefined labels)
          - acc: mean correctness over all label entries
    """
    assert y_true.shape == y_score.shape, (y_true.shape, y_score.shape)
    y_true = y_true.astype(np.int32)

    per_label_auc: Dict[int, float] = {}
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        ys = y_score[:, i]
        # roc_auc_score fails if only one class is present
        if len(np.unique(yt)) < 2:
            continue
        a = float(roc_auc_score(yt, ys))
        per_label_auc[i] = a
        aucs.append(a)

    auc = float(np.mean(aucs)) if len(aucs) else float("nan")

    y_pred = (y_score >= threshold).astype(np.int32)
    acc = float((y_pred == y_true).mean())

    return Metrics(auc=auc, acc=acc, per_label_auc=per_label_auc)
