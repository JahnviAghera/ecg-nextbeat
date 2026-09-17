#!/usr/bin/env python3
"""Deliverable 4 — external baseline reimplemented and retrained on Protocol B.

Architecture: the residual 1-D CNN of Kachuee, Fazeli & Sarrafzadeh (2018),
"ECG Heartbeat Classification: A Deep Transferable Representation" (ICHI 2018).

Why this one. Of the candidates in the literature review it is the most
completely specified: the paper gives the full layer table (initial Conv1D(32,5),
five residual blocks of two Conv1D(32,5) with a skip connection and
max-pooling(5, stride 2), then two dense layers), uses no proprietary or
undisclosed preprocessing, and its reference implementation has been reproduced
widely enough that the architecture is unambiguous. Reimplementing a model whose
preprocessing is under-described would confound the architecture with a guess at
its front end.

What was changed, and why. The original classifies a single 187-sample beat that
has already occurred. The task here is to predict the class of the beat that
follows a three-beat window. The backbone is therefore wrapped in
TimeDistributed so it encodes each of the three input beats, the three
representations are concatenated, and the softmax predicts beat i+1. Filter
counts, kernel size, depth, pooling and the dense head are unchanged. Beat
windows are 180 samples rather than 187, matching this paper's preprocessing so
the comparison isolates the architecture.

This is a lower bound on what the published model can do, not a refutation of
it: it was designed for retrospective classification and is being asked to do
something else.

    python run_external_baseline.py --data ../data --out ../revision/exports
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgnextbeat import data as datalib               # noqa: E402
from ecgnextbeat import models, train                 # noqa: E402
from ecgnextbeat.config import CLASS_NAMES, EPOCHS, HEADLINE_SEED  # noqa: E402
from ecgnextbeat.metrics import per_class             # noqa: E402

PROBA_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./revision/exports")
    ap.add_argument("--seed", type=int, default=HEADLINE_SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    partition = datalib.build_partition("B", args.data, args.seed)
    print(f"Protocol B: train {partition.X_train.shape[0]:,} / "
          f"val {partition.X_val.shape[0]:,} / test {partition.X_test.shape[0]:,}")

    result = train.train_and_evaluate(
        models.build_external_baseline, partition, seed=args.seed,
        epochs=args.epochs,
        checkpoint_dir=os.path.join(args.out, "external_baseline"),
        verbose=args.verbose)

    print(f"\n{result.parameters:,} parameters, {result.epochs_trained} epochs, "
          f"{result.seconds / 60:.1f} min")
    for key, value in result.metrics.items():
        print(f"  {key:20s} {value:.6f}")

    # Same schema as the other prediction exports, so the files join directly.
    frame = pd.DataFrame({
        "row_index": np.arange(len(partition.y_test), dtype=int),
        "sequence_id": [f"{r}_{w:05d}" for r, w in
                        zip(partition.test_record_ids, partition.test_window_starts)],
        "record_id": partition.test_record_ids,
        "window_start": partition.test_window_starts,
        "y_true": partition.y_test,
        "y_true_label": [CLASS_NAMES[c] for c in partition.y_test],
        "y_prev": partition.test_y_prev,
        "y_prev_label": [CLASS_NAMES[c] for c in partition.test_y_prev],
        "y_pred": result.y_pred,
        "y_pred_label": [CLASS_NAMES[c] for c in result.y_pred],
        "y_pred_conf": result.y_proba.max(axis=1),
    })
    for j, column in enumerate(PROBA_COLUMNS):
        frame[column] = result.y_proba[:, j]

    predictions_path = os.path.join(
        args.out, "external_baseline_protocolB_predictions.csv")
    frame.to_csv(predictions_path, index=False)

    per_class_path = os.path.join(args.out, "external_baseline_per_class.csv")
    per_class(partition.y_test, result.y_pred).to_csv(per_class_path, index=False)

    note_path = os.path.join(args.out, "external_baseline_note.md")
    with open(note_path, "w") as fh:
        fh.write("# External baseline\n\n")
        fh.write(__doc__.split("    python")[0].strip() + "\n\n")
        fh.write(f"## Result (Protocol B, seed {args.seed})\n\n")
        fh.write(f"- Parameters: {result.parameters:,}\n")
        fh.write(f"- Epochs trained: {result.epochs_trained}\n")
        fh.write(f"- Test sequences: {len(partition.y_test):,}\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        for key, value in result.metrics.items():
            fh.write(f"| {key} | {value:.6f} |\n")

    metrics_path = os.path.join(args.out, "external_baseline_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump({"seed": args.seed, "parameters": result.parameters,
                   "epochs_trained": result.epochs_trained,
                   "metrics": result.metrics}, fh, indent=2)

    for path in (predictions_path, per_class_path, note_path, metrics_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
