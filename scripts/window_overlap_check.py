#!/usr/bin/env python3
"""
window_overlap_check.py — beat-window overlap / leakage check (deliverable 3).

Every beat is a 180-sample window centred on its annotated R-peak, spanning
[R - 90, R + 90). Beat i's window therefore overlaps beat i+1's window exactly
when

    R_i + 90  >  R_{i+1} - 90        i.e.   RR_i < 180 samples

which at 360 Hz is an RR interval below 0.500 s, a rate above 120 bpm.

This matters because the paper's task is to predict beat i+1 from beats
i-2, i-1, i. If the last input beat's window already contains samples from the
target beat's window, the "not-yet-observed" beat has been partly observed.

Two neighbour definitions are reported, because they answer different questions:

  retained   the next beat that survives the pipeline's class filter and
             boundary rule — this is the neighbour relation the sequence
             constructor actually uses, so it is the one that governs leakage
             in the trained models. Written to the CSV.
  annotated  the next annotated beat of any symbol, including the paced and
             artefact beats the pipeline discards — the physically correct
             question of whether the window runs into the next depolarization.
             Reported to stdout and in the *_detail.csv.

A beat with no successor (the last in its record) is excluded from the
denominator.

Usage
-----
    python window_overlap_check.py --data /path/to/mit-bih/data \\
                                   --out ./revision/exports
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

BEAT_LENGTH = 180
SMOOTH_WINDOW = 5

CLASS_MAP = {"N": 0, "L": 1, "R": 2, "A": 3, "e": 4, "a": 5, "J": 6, "j": 7}
CLASS_NAMES = ["N", "LBBB", "RBBB", "APC", "AESC", "ABERR", "NPC", "NESC"]
NUM_CLASSES = 8

TRAIN_RECORDS = ["100", "101", "103", "105", "108", "109", "111", "113", "115",
                 "116", "117", "121", "122", "123", "124", "200", "203", "205",
                 "207", "208", "209", "213", "214", "215", "217", "219", "220",
                 "221", "222", "223", "228", "230", "231", "233", "234"]
VAL_RECORDS = ["106", "112", "119", "202", "210", "212", "232"]
TEST_RECORDS = ["102", "104", "107", "114", "118", "201"]
ALL_RECORDS = TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS


def smooth(signal: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return np.convolve(signal, np.ones(window) / window, mode="same")


def record_beats(data_path: str, record: str) -> pd.DataFrame:
    """Every annotated beat of a record, flagged for pipeline retention.

    The retention rule is the pipeline's: symbol in CLASS_MAP and the full
    180-sample window inside the signal.
    """
    path = os.path.join(data_path, record)
    try:
        n = len(smooth(wfdb.rdrecord(path).p_signal[:, 0]))
        fs = float(wfdb.rdrecord(path).fs)
    except Exception:
        header = wfdb.rdheader(path)
        n, fs = header.sig_len, float(header.fs)

    ann = wfdb.rdann(path, "atr")
    half = BEAT_LENGTH // 2
    peaks = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol)

    in_class = np.isin(symbols, list(CLASS_MAP))
    in_bounds = (peaks - half >= 0) & (peaks + half <= n)

    return pd.DataFrame({
        "record": record,
        "rpeak": peaks,
        "symbol": symbols,
        "retained": in_class & in_bounds,
        "class_index": [CLASS_MAP.get(s, -1) for s in symbols],
        "fs": fs,
    })


def overlap_flags(peaks: np.ndarray) -> np.ndarray:
    """True where this beat's window runs past the start of the next one's.

    The last element is False and is masked out by the caller — it has no
    successor to overlap.
    """
    half = BEAT_LENGTH // 2
    flags = np.zeros(len(peaks), dtype=bool)
    if len(peaks) > 1:
        flags[:-1] = (peaks[:-1] + half) > (peaks[1:] - half)
    return flags


def annotate(beats: pd.DataFrame) -> pd.DataFrame:
    """Add overlap flags under both neighbour definitions, per record."""
    frames = []
    for record, group in beats.groupby("record", sort=False):
        group = group.sort_values("rpeak").reset_index(drop=True)

        # Neighbour = next annotated beat of any symbol.
        group["overlap_annotated"] = overlap_flags(group["rpeak"].to_numpy())
        group["has_next_annotated"] = np.arange(len(group)) < len(group) - 1

        # Neighbour = next beat the pipeline keeps.
        group["overlap_retained"] = False
        group["has_next_retained"] = False
        kept = group.index[group["retained"]].to_numpy()
        if len(kept) > 1:
            flags = overlap_flags(group.loc[kept, "rpeak"].to_numpy())
            group.loc[kept, "overlap_retained"] = flags
            group.loc[kept[:-1], "has_next_retained"] = True

        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def summarise(beats: pd.DataFrame, partition: str,
              flag: str, has_next: str) -> pd.DataFrame:
    """Per-class overlap counts over the retained beats of a partition."""
    subset = beats[beats["retained"] & beats[has_next]]
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        selected = subset[subset["class_index"] == index]
        total = len(selected)
        overlapping = int(selected[flag].sum())
        rows.append({
            "partition": partition,
            "class": name,
            "total_beats": total,
            "overlapping_beats": overlapping,
            "pct_overlapping": round(100.0 * overlapping / total, 4) if total else 0.0,
        })
    total = len(subset)
    overlapping = int(subset[flag].sum())
    rows.append({
        "partition": partition,
        "class": "ALL",
        "total_beats": total,
        "overlapping_beats": overlapping,
        "pct_overlapping": round(100.0 * overlapping / total, 4) if total else 0.0,
    })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./revision/exports")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frames = []
    for record in ALL_RECORDS:
        try:
            frames.append(record_beats(args.data, record))
        except Exception as exc:
            print(f"[ERROR] {record}: {exc}")
    beats = annotate(pd.concat(frames, ignore_index=True))

    print(f"Records read: {beats['record'].nunique()}")
    print(f"Annotated beats: {len(beats):,}")
    print(f"Retained by the pipeline: {int(beats['retained'].sum()):,}")
    fs = sorted(beats["fs"].unique())
    print(f"Sampling rate(s): {fs} Hz -> "
          f"180 samples = {180 / fs[0]:.3f} s, overlap below {60 * fs[0] / 180:.0f} bpm\n")

    is_test = beats["record"].isin(TEST_RECORDS)

    # Requested deliverable: retained-neighbour definition, two partitions.
    table = pd.concat([
        summarise(beats, "full_corpus", "overlap_retained", "has_next_retained"),
        summarise(beats[is_test], "protocolB_test", "overlap_retained", "has_next_retained"),
    ], ignore_index=True)
    path = os.path.join(args.out, "window_overlap_by_class.csv")
    table.to_csv(path, index=False)

    # Detail file: both neighbour definitions side by side.
    detail = pd.concat([
        summarise(beats, "full_corpus", "overlap_retained", "has_next_retained")
        .assign(neighbour="next_retained_beat"),
        summarise(beats[is_test], "protocolB_test", "overlap_retained", "has_next_retained")
        .assign(neighbour="next_retained_beat"),
        summarise(beats, "full_corpus", "overlap_annotated", "has_next_annotated")
        .assign(neighbour="next_annotated_beat"),
        summarise(beats[is_test], "protocolB_test", "overlap_annotated", "has_next_annotated")
        .assign(neighbour="next_annotated_beat"),
    ], ignore_index=True)
    detail_path = os.path.join(args.out, "window_overlap_by_class_detail.csv")
    detail.to_csv(detail_path, index=False)

    print(table.to_string(index=False))
    print(f"\nwrote {path}")
    print(f"wrote {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
