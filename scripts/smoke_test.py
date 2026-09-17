#!/usr/bin/env python3
"""End-to-end smoke test: every stage of the pipeline on a small subset.

Exercises imports, beat extraction, sequence construction, both protocols'
split logic, every model builder, the budget fitter, a short training run and
the metric suite. Trains for two epochs on a handful of records, so it finishes
in a couple of minutes and proves the repository is wired correctly -- it does
NOT reproduce any published number.

    python smoke_test.py --data ../../data
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append((name, True, detail))
            print(f"  [ok]   {name}: {detail}")
        except Exception as exc:                      # noqa: BLE001
            checks.append((name, False, repr(exc)))
            print(f"  [FAIL] {name}: {exc!r}")

    print("ecg-nextbeat smoke test\n")

    from ecgnextbeat import config, data, metrics, models, train

    check("imports", lambda: f"version {__import__('ecgnextbeat').__version__}")
    check("record split disjoint", lambda: (
        "35/7/6" if (set(config.TRAIN_RECORDS).isdisjoint(config.VAL_RECORDS)
                     and set(config.TRAIN_RECORDS).isdisjoint(config.TEST_RECORDS)
                     and set(config.VAL_RECORDS).isdisjoint(config.TEST_RECORDS)
                     and len(config.ALL_RECORDS) == 48)
        else (_ for _ in ()).throw(AssertionError("record sets overlap"))))

    def beats():
        b, labels, peaks = data.extract_beats(args.data, "100")
        assert b.shape[1] == config.BEAT_LENGTH
        assert len(b) == len(labels) == len(peaks)
        return f"record 100 -> {len(b):,} beats of {b.shape[1]} samples"
    check("beat extraction", beats)

    def sequences():
        b, labels, _ = data.extract_beats(args.data, "100")
        X, y, y_prev, starts = data.build_sequences(b, labels)
        # The target must be the beat AFTER the window, never inside it.
        assert np.array_equal(y, labels[config.SEQUENCE_LENGTH:])
        assert np.array_equal(y_prev, labels[config.SEQUENCE_LENGTH - 1:-1])
        assert len(X) == len(labels) - config.SEQUENCE_LENGTH
        return f"{len(X):,} windows, target = labels[i+{config.SEQUENCE_LENGTH}]"
    check("sequence construction (target is outside the window)", sequences)

    def builders():
        sizes = {}
        for name, fn in models.BUILDERS.items():
            sizes[name] = fn().count_params()
        assert sizes["model1"] == 1_577_832, sizes["model1"]
        assert sizes["model2"] == 3_667_656, sizes["model2"]
        return ", ".join(f"{k} {v:,}" for k, v in sizes.items())
    check("model builders match published parameter counts", builders)

    def budgets():
        out = []
        for variant in models.VARIANTS:
            _, params = models.fit_to_budget(variant, 1_800_000, 0.10)
            out.append(params)
        spread = (max(out) - min(out)) / min(out)
        assert spread < 0.10, spread
        return f"4 variants within {100 * spread:.2f}% of each other"
    check("matched parameter budgets", budgets)

    def short_train():
        # A tiny Protocol-B-shaped partition from three records.
        X, y, y_prev, record_ids, starts = data.process_records(
            args.data, ["100", "101", "103"])
        X = X.reshape(-1, config.SEQUENCE_LENGTH, config.BEAT_LENGTH, 1)
        split = len(y) // 2
        partition = data.Partition(
            X[:split], y[:split], X[split:], y[split:], X[split:], y[split:],
            record_ids[split:], y_prev[split:], starts[split:])
        result = train.train_and_evaluate(models.build_model1, partition,
                                          seed=42, epochs=2, verbose=0)
        assert result.y_proba.shape == (len(partition.y_test), config.NUM_CLASSES)
        assert np.allclose(result.y_proba.sum(axis=1), 1.0, atol=1e-4)
        return (f"2 epochs on {split:,} sequences, "
                f"accuracy {result.metrics['accuracy']:.3f}")
    check("training + inference", short_train)

    def metric_suite():
        y_true = np.array([0, 1, 2, 0, 5])
        out = metrics.all_metrics(y_true, y_true)
        assert out["accuracy"] == 1.0
        # Macro-F1 averages over all eight classes, so a perfect prediction
        # over four present classes gives 4/8, not 1.0.
        assert abs(out["macro_f1"] - 0.5) < 1e-9, out["macro_f1"]
        return "macro-average denominator is all 8 classes"
    check("metric conventions", metric_suite)

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
