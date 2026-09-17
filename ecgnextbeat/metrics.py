"""The metric suite. Macro-averages are taken over ALL eight defined classes.

Averaging instead over only the classes present in y_true or y_pred makes the
denominator depend on the predictions and inflates macro-F1; see Section 3.6 of
the revision and VERIFICATION.md discrepancy 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

from .config import CLASS_NAMES, NUM_CLASSES


def all_metrics(y_true, y_pred) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(pr.mean()),
        "macro_recall": float(rc.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def per_class(y_true, y_pred):
    import pandas as pd

    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    rows = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rows.append({
            "class": CLASS_NAMES[i], "support": int(sup[i]),
            "precision": pr[i], "sensitivity": rc[i],
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "f1": f1[i], "tp": int(tp), "fp": int(fp), "fn": int(fn),
        })
    return pd.DataFrame(rows)


def summarise(frame, group_columns, metric_columns):
    """mean +/- SD over seeds, formatted for the manuscript."""
    aggregated = frame.groupby(group_columns)[metric_columns].agg(["mean", "std"])
    out = {}
    for metric in metric_columns:
        mean = aggregated[(metric, "mean")]
        std = aggregated[(metric, "std")].fillna(0.0)
        out[metric] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(mean, std)]
    import pandas as pd
    return pd.DataFrame(out, index=aggregated.index).reset_index()
