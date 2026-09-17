#!/usr/bin/env python3
"""Deliverable 1 — multi-seed replication of the headline numbers.

Retrains Model-I and Model-II under both protocols with several seeds. Only the
seed changes: architecture, hyperparameters and the split logic are identical
across runs. Under Protocol B the record split is fixed by Appendix A2, so the
seed affects only weight initialisation, shuffling and oversampling. Under
Protocol A the seed also moves the sequence-level split, which is the intended
behaviour -- resampling the split is part of what the replication measures.

Results append to multiseed_results.csv after every run, so an interrupted
sweep keeps what it finished.

    python run_multiseed.py --data ../data --out ../revision/exports
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgnextbeat import data as datalib          # noqa: E402
from ecgnextbeat import models, train            # noqa: E402
from ecgnextbeat.config import EPOCHS, REPLICATION_SEEDS  # noqa: E402
from ecgnextbeat.metrics import summarise        # noqa: E402

METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "macro_f1",
                  "weighted_f1", "mcc", "kappa"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./revision/exports")
    ap.add_argument("--seeds", type=int, nargs="+", default=REPLICATION_SEEDS)
    ap.add_argument("--protocols", nargs="+", default=["A", "B"])
    ap.add_argument("--models", nargs="+", default=["model1", "model2"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "multiseed_results.csv")
    rows = (pd.read_csv(results_path).to_dict("records")
            if os.path.exists(results_path) else [])
    done = {(r["protocol"], r["model"], r["seed"]) for r in rows}

    for protocol in args.protocols:
        for seed in args.seeds:
            # The partition depends on the seed under Protocol A (split) and
            # under Protocol B (oversampling only), so rebuild it per seed.
            partition = None
            for model_name in args.models:
                if (protocol, model_name, seed) in done:
                    print(f"[skip] {protocol}/{model_name}/seed {seed} already done")
                    continue
                if partition is None:
                    partition = datalib.build_partition(protocol, args.data, seed)
                    print(f"\nProtocol {protocol}, seed {seed}: "
                          f"train {partition.X_train.shape[0]:,} / "
                          f"val {partition.X_val.shape[0]:,} / "
                          f"test {partition.X_test.shape[0]:,}")

                print(f"  training {model_name} (seed {seed}) ...", flush=True)
                result = train.train_and_evaluate(
                    models.BUILDERS[model_name], partition, seed=seed,
                    epochs=args.epochs, verbose=args.verbose)

                row = {"protocol": protocol, "model": model_name, "seed": seed,
                       **{k: result.metrics[k] for k in METRIC_COLUMNS},
                       "parameters": result.parameters,
                       "epochs_trained": result.epochs_trained,
                       "seconds": round(result.seconds, 1)}
                rows.append(row)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                print(f"    acc {row['accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}  "
                      f"({result.epochs_trained} epochs, {result.seconds / 60:.1f} min)")

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("nothing to summarise")
        return 1

    frame[["protocol", "model", "seed"] + METRIC_COLUMNS + [
        "parameters", "epochs_trained", "seconds"]].to_csv(results_path, index=False)

    summary = summarise(frame, ["protocol", "model"], METRIC_COLUMNS)
    summary_path = os.path.join(args.out, "multiseed_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nmean +/- SD over seeds:")
    print(summary.to_string(index=False))
    print(f"\nwrote {results_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
