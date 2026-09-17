#!/usr/bin/env python3
"""
record_level_bootstrap.py — per-record bootstrap intervals under Protocol B.

Section 4.8 of the revision discusses record-level variability in prose and
Table 9 gives per-record point estimates. This produces the interval version:
for each system and each test record, a 2,000-sample percentile bootstrap over
the sequences of that record, matching the convention used everywhere else in
the paper (2,000 resamples, 95% percentile interval, seed 42).

Reads the exported per-sequence prediction files, so it recomputes nothing about
the models themselves.

Caveat worth reading before quoting these: resampling within a record cannot
widen an interval past the variability the record contains. Records 102 and 104
are small (96 and 160 sequences) and almost purely normal, so a system that is
perfect on them gets the degenerate interval [1.000, 1.000] — which reflects the
absence of errors to resample, not confidence in the estimate. Such rows are
flagged in the `degenerate` column.

Usage
-----
    python record_level_bootstrap.py --exports ./revision/exports \\
                                     --out ./revision/exports
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, f1_score,
                             matthews_corrcoef, cohen_kappa_score)

NUM_CLASSES = 8
CLASS_NAMES = ["N", "LBBB", "RBBB", "APC", "AESC", "ABERR", "NPC", "NESC"]

SYSTEMS = {
    "model1": "Model-I (CNN-LSTM)",
    "model2": "Model-II (CNN-BiLSTM-Transformer)",
    "persistence": "Persistence baseline",
    "majority": "Majority-class baseline",
}

N_BOOTSTRAP = 2000
SEED = 42


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over all eight defined classes (the revision's convention)."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    return float(f1.mean())


METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "macro_f1": macro_f1,
    "weighted_f1": lambda a, b: f1_score(a, b, average="weighted", zero_division=0),
    "mcc": matthews_corrcoef,
    "kappa": cohen_kappa_score,
}


def bootstrap(y_true: np.ndarray, y_pred: np.ndarray, fn,
              n: int = N_BOOTSTRAP, seed: int = SEED):
    """Percentile bootstrap interval for one metric on one subset."""
    rng = np.random.default_rng(seed)
    size = len(y_true)
    values = np.empty(n)
    for i in range(n):
        sample = rng.integers(0, size, size)
        values[i] = fn(y_true[sample], y_pred[sample])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", default="./revision/exports",
                    help="directory holding *_protocolB_predictions.csv")
    ap.add_argument("--out", default="./revision/exports")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frames = {}
    for key in SYSTEMS:
        path = os.path.join(args.exports, f"{key}_protocolB_predictions.csv")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        frames[key] = pd.read_csv(path)

    if not frames:
        print("No prediction files found — run export_protocolB_predictions.py first.")
        return 1

    reference = next(iter(frames.values()))
    records = sorted(reference["record_id"].unique())

    rows = []
    for key, frame in frames.items():
        for record in list(records) + ["ALL"]:
            mask = (slice(None) if record == "ALL"
                    else (frame["record_id"] == record).to_numpy())
            subset = frame if record == "ALL" else frame[mask]
            y_true = subset["y_true"].to_numpy()
            y_pred = subset["y_pred"].to_numpy()

            row = {
                "system": SYSTEMS[key],
                "system_key": key,
                "record": record,
                "n_sequences": len(subset),
                "n_classes_present": int(len(np.unique(y_true))),
            }
            for name, fn in METRICS.items():
                point = float(fn(y_true, y_pred))
                low, high = bootstrap(y_true, y_pred, fn, n=args.bootstrap)
                row[name] = point
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
            row["degenerate"] = bool(
                row["accuracy_ci_low"] == row["accuracy_ci_high"])
            rows.append(row)

    table = pd.DataFrame(rows)
    path = os.path.join(args.out, "record_level_bootstrap.csv")
    table.to_csv(path, index=False)

    # Compact rendering for the manuscript.
    display = table.assign(
        Accuracy=lambda d: d.apply(
            lambda r: f"{r.accuracy:.3f} [{r.accuracy_ci_low:.3f}, {r.accuracy_ci_high:.3f}]",
            axis=1),
        Macro_F1=lambda d: d.apply(
            lambda r: f"{r.macro_f1:.3f} [{r.macro_f1_ci_low:.3f}, {r.macro_f1_ci_high:.3f}]",
            axis=1),
    )[["system", "record", "n_sequences", "Accuracy", "Macro_F1", "degenerate"]]
    display_path = os.path.join(args.out, "record_level_bootstrap_display.csv")
    display.to_csv(display_path, index=False)

    print(display.to_string(index=False))
    print(f"\nwrote {path}")
    print(f"wrote {display_path}")
    degenerate = int(table["degenerate"].sum())
    if degenerate:
        print(f"\n{degenerate} of {len(table)} rows have a degenerate accuracy interval "
              f"(no errors, or no correct predictions, left to resample).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
