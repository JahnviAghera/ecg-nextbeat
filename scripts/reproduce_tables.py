#!/usr/bin/env python3
"""Reproduce Tables 3, 5, 6, 7 and 8 of the manuscript from scratch.

    table3  Per-class F1 under Protocol A, both models.
    table5  Per-class F1 under Protocol B, both models.
    table6  Architecture ablation under Protocol B, four variants at their
            ORIGINAL (unmatched) sizes. For the matched-budget version see
            run_matched_ablation.py.
    table7  Model-II under Protocol B with input windows of 1, 2, 3, 5, 10 beats.
    table8  Model-II on the Protocol B test partition with the three input beats
            randomly permuted within each sequence -- the temporal-order sanity
            check. Trains once, then evaluates ordered and permuted.

Every table uses seed 42, the seed behind the headline single-run results.

    python reproduce_tables.py --data ../data --out ../revision/exports --table all
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecgnextbeat import data as datalib                      # noqa: E402
from ecgnextbeat import models, train                        # noqa: E402
from ecgnextbeat.config import CLASS_NAMES, EPOCHS, HEADLINE_SEED  # noqa: E402
from ecgnextbeat.metrics import all_metrics, per_class       # noqa: E402

SEQUENCE_LENGTHS = [1, 2, 3, 5, 10]


def _per_class_table(protocol: str, args) -> pd.DataFrame:
    partition = datalib.build_partition(protocol, args.data, args.seed)
    print(f"Protocol {protocol}: train {partition.X_train.shape[0]:,} / "
          f"test {partition.X_test.shape[0]:,}")
    frames = []
    for name in ("model1", "model2"):
        print(f"  training {name} ...", flush=True)
        result = train.train_and_evaluate(models.BUILDERS[name], partition,
                                          seed=args.seed, epochs=args.epochs,
                                          verbose=args.verbose)
        frame = per_class(partition.y_test, result.y_pred)
        frame.insert(0, "model", name)
        frame.insert(0, "protocol", protocol)
        frames.append(frame)
        print(f"    accuracy {result.metrics['accuracy']:.4f}")
    return pd.concat(frames, ignore_index=True)


def table3(args):
    return _per_class_table("A", args)


def table5(args):
    return _per_class_table("B", args)


def table6(args):
    partition = datalib.build_partition("B", args.data, args.seed)
    rows = []
    for variant in models.VARIANTS:
        print(f"  training {variant} ...", flush=True)
        build_fn = functools.partial(models.build_variant, variant, 1.0)
        result = train.train_and_evaluate(build_fn, partition, seed=args.seed,
                                          epochs=args.epochs, verbose=args.verbose)
        rows.append({"variant": variant, "parameters": result.parameters,
                     **result.metrics})
        print(f"    accuracy {result.metrics['accuracy']:.4f}")
    return pd.DataFrame(rows)


def table7(args):
    rows = []
    for length in SEQUENCE_LENGTHS:
        print(f"  sequence length {length} ...", flush=True)
        partition = datalib.build_partition("B", args.data, args.seed,
                                            sequence_length=length)
        build_fn = functools.partial(models.build_model2, sequence_length=length)
        result = train.train_and_evaluate(build_fn, partition, seed=args.seed,
                                          epochs=args.epochs, verbose=args.verbose)
        rows.append({"sequence_length": length,
                     "test_sequences": int(len(partition.y_test)),
                     "parameters": result.parameters, **result.metrics})
        print(f"    accuracy {result.metrics['accuracy']:.4f}")
    return pd.DataFrame(rows)


def table8(args):
    """Train once, evaluate on ordered and on per-sequence permuted inputs."""
    import tensorflow as tf
    from tensorflow.keras import optimizers

    partition = datalib.build_partition("B", args.data, args.seed)
    train.set_seed(args.seed)
    model = models.build_model2()
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    from tensorflow.keras import callbacks
    model.fit(partition.X_train, partition.y_train,
              validation_data=(partition.X_val, partition.y_val),
              epochs=args.epochs, batch_size=64, verbose=args.verbose,
              callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                 restore_best_weights=True)])

    rng = np.random.default_rng(args.seed)
    shuffled = partition.X_test.copy()
    for i in range(len(shuffled)):
        shuffled[i] = shuffled[i][rng.permutation(shuffled.shape[1])]

    rows = []
    for label, X in (("ordered", partition.X_test), ("permuted", shuffled)):
        y_pred = np.asarray(model.predict(X, batch_size=256, verbose=0)).argmax(1)
        rows.append({"input_order": label, **all_metrics(partition.y_test, y_pred)})
        print(f"  {label}: accuracy {rows[-1]['accuracy']:.4f}")
    tf.keras.backend.clear_session()
    return pd.DataFrame(rows)


TABLES = {"table3": table3, "table5": table5, "table6": table6,
          "table7": table7, "table8": table8}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./revision/exports")
    ap.add_argument("--table", nargs="+", default=["all"],
                    choices=list(TABLES) + ["all"])
    ap.add_argument("--seed", type=int, default=HEADLINE_SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wanted = list(TABLES) if "all" in args.table else args.table

    for name in wanted:
        print(f"\n=== {name} ===")
        frame = TABLES[name](args)
        path = os.path.join(args.out, f"{name}_reproduction.csv")
        frame.to_csv(path, index=False)
        print(frame.to_string(index=False))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
