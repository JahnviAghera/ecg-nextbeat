#!/usr/bin/env python3
"""Deliverable 2 — matched-parameter-budget architecture ablation.

The published ablation compares four variants whose parameter counts differ by
more than 4x (CNN-only 836,808 to CNN+BiLSTM+Transformer 3,667,656), so any
difference between them confounds the temporal mechanism with capacity. This
re-runs the same four variants with every non-convolutional width rescaled so
all four land within +/-10% of a common budget, defaulting to 1,800,000 -- the
size of the CNN+BiLSTM variant in the original ablation.

The convolutional feature extractor is held fixed at (64, 128, 256) for all
four, so the budget is matched by scaling the temporal block and dense head
only. Each variant is trained with several seeds under Protocol B.

    python run_matched_ablation.py --data ../data --out ../revision/exports
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgnextbeat import data as datalib          # noqa: E402
from ecgnextbeat import models, train            # noqa: E402
from ecgnextbeat.config import ABLATION_SEEDS, EPOCHS  # noqa: E402
from ecgnextbeat.metrics import summarise        # noqa: E402

METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "macro_f1",
                  "weighted_f1", "mcc", "kappa"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./revision/exports")
    ap.add_argument("--target", type=int, default=1_800_000)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="+", default=ABLATION_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Fitting each variant to {args.target:,} parameters "
          f"(+/-{args.tolerance:.0%}):")
    widths = {}
    for variant in models.VARIANTS:
        width, params = models.fit_to_budget(variant, args.target, args.tolerance)
        widths[variant] = (width, params)
        delta = 100.0 * (params - args.target) / args.target
        print(f"  {variant:24s} width x{width:.4f} -> {params:,} ({delta:+.1f}%)")

    budget_frame = pd.DataFrame(
        [{"variant": v, "width_multiplier": w, "parameters": p,
          "target": args.target,
          "pct_from_target": 100.0 * (p - args.target) / args.target}
         for v, (w, p) in widths.items()])
    budget_path = os.path.join(args.out, "matched_ablation_budgets.csv")
    budget_frame.to_csv(budget_path, index=False)

    results_path = os.path.join(args.out, "matched_ablation_results.csv")
    rows = (pd.read_csv(results_path).to_dict("records")
            if os.path.exists(results_path) else [])
    done = {(r["variant"], r["seed"]) for r in rows}

    for seed in args.seeds:
        partition = datalib.build_partition("B", args.data, seed)
        print(f"\nProtocol B, seed {seed}: train {partition.X_train.shape[0]:,}")
        for variant in models.VARIANTS:
            if (variant, seed) in done:
                print(f"[skip] {variant}/seed {seed} already done")
                continue
            width, params = widths[variant]
            print(f"  training {variant} ({params:,} params, seed {seed}) ...",
                  flush=True)
            build_fn = functools.partial(models.build_variant, variant, width)
            result = train.train_and_evaluate(build_fn, partition, seed=seed,
                                              epochs=args.epochs,
                                              verbose=args.verbose)
            rows.append({"variant": variant, "parameters": result.parameters,
                         "seed": seed,
                         **{k: result.metrics[k] for k in METRIC_COLUMNS},
                         "epochs_trained": result.epochs_trained,
                         "seconds": round(result.seconds, 1)})
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"    acc {rows[-1]['accuracy']:.4f}  "
                  f"macro-F1 {rows[-1]['macro_f1']:.4f}  "
                  f"({result.seconds / 60:.1f} min)")

    frame = pd.DataFrame(rows)
    if frame.empty:
        return 1
    summary = summarise(frame, ["variant"], METRIC_COLUMNS)
    summary_path = os.path.join(args.out, "matched_ablation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nmean +/- SD over seeds:")
    print(summary.to_string(index=False))
    print(f"\nwrote {budget_path}\nwrote {results_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
