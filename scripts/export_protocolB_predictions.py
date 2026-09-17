#!/usr/bin/env python3
"""
export_protocolB_predictions.py — raw per-sequence predictions for Protocol B.

Rebuilds the Protocol B (record-wise, inter-patient) test partition from the raw
MIT-BIH WFDB files, runs the stored Model-I and Model-II checkpoints over it, and
writes one row per test sequence for each system. No metrics are computed and
nothing is aggregated; the files are intended for external analysis.

Systems exported
----------------
    model1       CNN-LSTM                      (record_wise/model_I/best_model.keras)
    model2       CNN-BiLSTM-Transformer        (record_wise/model_II/best_model.keras)
    persistence  y_pred = y_prev               (no model)
    majority     y_pred = N (class 0) always   (no model)

Every file has identical row order and identical `sequence_id` values, so they can
be joined or compared directly (e.g. pandas merge on `sequence_id`, or plain
row-wise comparison).

Row order is: records in the fixed order 102, 104, 107, 114, 118, 201; within a
record, windows in temporal order. This is the same order the training pipeline
used, so the exported `y_pred` columns line up with the stored
`*_recordwise_predictions.npy` vectors — `--verify` checks that explicitly.

Usage
-----
    python export_protocolB_predictions.py \
        --data   /path/to/mit-bih/data \
        --models /path/to/data/saved_models \
        --out    ./revision/exports

Columns
-------
    row_index        0-based position in the partition (join key, integer)
    sequence_id      "<record>_<window index within record>", e.g. "102_00000"
    record_id        source record number as an integer, e.g. 102
    window_start     index of the first input beat within the record's retained beats
    y_true           class index of the target beat x_{i+1}
    y_true_label     AAMI abbreviation of y_true
    y_prev           class index of the last observed input beat x_i
    y_prev_label     AAMI abbreviation of y_prev
    y_pred           predicted class index
    y_pred_label     AAMI abbreviation of y_pred
    y_pred_conf      top-1 probability
    proba_N ... proba_NESC    full 8-class probability vector

For the two baselines the probability vector is the one-hot encoding of their
deterministic prediction (confidence 1.0); they are not calibrated probabilities.

A companion file `protocolB_sequence_geometry.csv` is written alongside the
prediction files, joinable on `sequence_id` (or `row_index`). It carries the
R-peak sample index of each input beat and of the target beat, the RR intervals
between them, and how far the last input beat's 180-sample window overlaps the
target beat's window — the inputs needed for a beat-window overlap / leakage
check, which the prediction files alone cannot support.

Class indices follow the corrected mapping: index 4 is AESC (symbol 'e') and
index 5 is ABERR (symbol 'a'). See revision/VERIFICATION.md, discrepancy 2.
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

# --------------------------------------------------------------------------
# Fixed experimental constants — these define the protocol and must not drift.
# Identical to revision/code/reproduce_revision.py.
# --------------------------------------------------------------------------
SEQUENCE_LENGTH = 3
BEAT_LENGTH = 180
SMOOTH_WINDOW = 5

CLASS_MAP = {"N": 0, "L": 1, "R": 2, "A": 3, "e": 4, "a": 5, "J": 6, "j": 7}
CLASS_NAMES = ["N", "LBBB", "RBBB", "APC", "AESC", "ABERR", "NPC", "NESC"]
NUM_CLASSES = 8

TEST_RECORDS = ["102", "104", "107", "114", "118", "201"]

PROBA_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]

# Checkpoints and the stored prediction vectors used by --verify.
MODELS = {
    "model1": {
        "label": "Model-I (CNN-LSTM)",
        "checkpoint": "record_wise/model_I/best_model.keras",
        "stored_predictions": "record_wise/model_I/Model_I_recordwise_predictions.npy",
    },
    "model2": {
        "label": "Model-II (CNN-BiLSTM-Transformer)",
        "checkpoint": "record_wise/model_II/best_model.keras",
        "stored_predictions": "record_wise/model_II/Model_II_recordwise_predictions.npy",
    },
}


# --------------------------------------------------------------------------
# Preprocessing — a reimplementation of RecordWiseECGProcessor from the
# training notebook, kept here so the export does not depend on the notebook.
# --------------------------------------------------------------------------
def smooth(signal: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return np.convolve(signal, np.ones(window) / window, mode="same")


def extract_beats(data_path: str, record: str):
    """Z-normalised 180-sample beat windows, class indices, and R-peak positions.

    A beat is retained iff its annotation symbol is in CLASS_MAP and its window
    lies entirely inside the signal. Order is temporal. The R-peak sample index
    of every retained beat is returned alongside it so that downstream analysis
    can reconstruct beat timing (RR intervals, window overlap) without going
    back to the WFDB files.
    """
    path = os.path.join(data_path, record)
    signal_record = wfdb.rdrecord(path)
    signal = smooth(signal_record.p_signal[:, 0])
    ann = wfdb.rdann(path, "atr")
    half = BEAT_LENGTH // 2

    beats, labels, peaks = [], [], []
    for peak, symbol in zip(ann.sample, ann.symbol):
        if symbol not in CLASS_MAP:
            continue
        if peak - half < 0 or peak + half > len(signal):
            continue
        beat = signal[peak - half: peak + half]
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        beats.append(beat)
        labels.append(CLASS_MAP[symbol])
        peaks.append(int(peak))

    return (np.asarray(beats, dtype=np.float32),
            np.asarray(labels, dtype=int),
            np.asarray(peaks, dtype=int),
            float(signal_record.fs))


def build_test_partition(data_path: str):
    """Reconstruct Protocol B: inputs, targets, previous-beat labels, provenance.

    Window i covers beats i, i+1, i+2; its target is beat i+3, so
        y_true[i] = labels[i + SEQUENCE_LENGTH]
        y_prev[i] = labels[i + SEQUENCE_LENGTH - 1]   (the last observed beat)
    """
    X, y_true, y_prev, record_ids, window_starts = [], [], [], [], []
    rpeaks_in, rpeaks_target, sampling_rates = [], [], []

    for record in TEST_RECORDS:
        beats, labels, peaks, fs = extract_beats(data_path, record)
        n = len(labels) - SEQUENCE_LENGTH
        if n <= 0:
            print(f"  {record}: 0 usable sequences "
                  f"({len(labels)} beats in the retained class set)")
            continue

        windows = np.stack([beats[i: i + SEQUENCE_LENGTH] for i in range(n)])
        X.append(windows)
        y_true.append(labels[SEQUENCE_LENGTH:])
        y_prev.append(labels[SEQUENCE_LENGTH - 1: -1])
        record_ids.append(np.full(n, int(record), dtype=int))
        window_starts.append(np.arange(n, dtype=int))
        rpeaks_in.append(np.stack([peaks[i: i + SEQUENCE_LENGTH] for i in range(n)]))
        rpeaks_target.append(peaks[SEQUENCE_LENGTH:])
        sampling_rates.append(np.full(n, fs, dtype=float))
        print(f"  {record}: {n:,} sequences")

    if not X:
        raise RuntimeError("No test sequences were generated — check --data.")

    return (
        np.concatenate(X).reshape(-1, SEQUENCE_LENGTH, BEAT_LENGTH, 1),
        np.concatenate(y_true),
        np.concatenate(y_prev),
        np.concatenate(record_ids),
        np.concatenate(window_starts),
        np.concatenate(rpeaks_in),
        np.concatenate(rpeaks_target),
        np.concatenate(sampling_rates),
    )


# --------------------------------------------------------------------------
# Beat geometry — R-peak positions, RR intervals and window overlap.
# --------------------------------------------------------------------------
def build_geometry(index: pd.DataFrame, rpeaks_in: np.ndarray,
                   rpeaks_target: np.ndarray, fs: np.ndarray) -> pd.DataFrame:
    """Per-sequence beat timing and input/target window overlap.

    Each beat is a 180-sample window centred on its R-peak, i.e. it spans
    [R - 90, R + 90). Consecutive windows therefore overlap whenever the RR
    interval is under 180 samples (0.5 s at 360 Hz, a rate above 120 bpm).

    `target_overlap_samples` is how many samples of the TARGET beat's window
    already appear inside the last input beat's window — the quantity that
    matters for whether the next-beat task leaks. `input_contains_target_rpeak`
    is the stronger condition that the input window extends past the target
    R-peak itself.
    """
    half = BEAT_LENGTH // 2
    prev_peak = rpeaks_in[:, -1]

    rr_prev_to_target = rpeaks_target - prev_peak
    overlap = np.maximum(0, (prev_peak + half) - (rpeaks_target - half))

    geometry = index[["row_index", "sequence_id", "record_id", "window_start"]].copy()
    geometry["fs_hz"] = fs
    for j in range(SEQUENCE_LENGTH):
        geometry[f"rpeak_input_{j}"] = rpeaks_in[:, j]
    geometry["rpeak_target"] = rpeaks_target
    for j in range(SEQUENCE_LENGTH - 1):
        geometry[f"rr_input_{j}_{j + 1}"] = rpeaks_in[:, j + 1] - rpeaks_in[:, j]
    geometry["rr_prev_to_target"] = rr_prev_to_target
    geometry["rr_prev_to_target_sec"] = rr_prev_to_target / fs
    geometry["input_window_start"] = rpeaks_in[:, 0] - half
    geometry["input_window_end"] = prev_peak + half
    geometry["target_window_start"] = rpeaks_target - half
    geometry["target_window_end"] = rpeaks_target + half
    geometry["target_overlap_samples"] = overlap
    geometry["target_overlap_fraction"] = overlap / BEAT_LENGTH
    geometry["input_target_overlap"] = overlap > 0
    geometry["input_contains_target_rpeak"] = (prev_peak + half) > rpeaks_target
    return geometry


# --------------------------------------------------------------------------
# Frame assembly
# --------------------------------------------------------------------------
def make_frame(index: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Attach a system's predictions to the shared per-sequence index."""
    y_pred = proba.argmax(axis=1)
    df = index.copy()
    df["y_pred"] = y_pred
    df["y_pred_label"] = [CLASS_NAMES[c] for c in y_pred]
    df["y_pred_conf"] = proba.max(axis=1)
    for j, column in enumerate(PROBA_COLUMNS):
        df[column] = proba[:, j]
    return df


def one_hot(y: np.ndarray) -> np.ndarray:
    proba = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    proba[np.arange(len(y)), y] = 1.0
    return proba


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="directory of MIT-BIH .hea/.atr/.dat files")
    ap.add_argument("--models", required=True, help="saved_models directory")
    ap.add_argument("--out", default="./revision/exports",
                    help="output directory (created if absent)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--npz", action="store_true",
                    help="also write a single .npz bundle of every array")
    ap.add_argument("--verify", action="store_true",
                    help="compare fresh predictions against the stored *.npy vectors")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Reconstructing the Protocol B test partition from raw annotations:")
    (X, y_true, y_prev, record_ids, window_starts,
     rpeaks_in, rpeaks_target, fs) = build_test_partition(args.data)
    n = len(y_true)
    print(f"  total: {n:,} sequences from {len(set(record_ids))} records\n")

    # Shared index — identical in every exported file, so the files join on
    # sequence_id (or on row_index, or row-wise without a key at all).
    index = pd.DataFrame({
        "row_index": np.arange(n, dtype=int),
        "sequence_id": [f"{r}_{w:05d}" for r, w in zip(record_ids, window_starts)],
        "record_id": record_ids,
        "window_start": window_starts,
        "y_true": y_true,
        "y_true_label": [CLASS_NAMES[c] for c in y_true],
        "y_prev": y_prev,
        "y_prev_label": [CLASS_NAMES[c] for c in y_prev],
    })
    assert index["sequence_id"].is_unique, "sequence_id is not unique"

    geometry = build_geometry(index, rpeaks_in, rpeaks_target, fs)

    frames, arrays, failures = {}, {}, []

    # ---- trained models --------------------------------------------------
    import keras  # imported late so the baselines still export without a GPU stack

    for key, spec in MODELS.items():
        path = os.path.join(args.models, spec["checkpoint"])
        if not os.path.exists(path):
            print(f"[skip] missing checkpoint: {path}")
            continue

        model = keras.models.load_model(path, compile=False)
        print(f"{spec['label']}: loaded {path} ({model.count_params():,} parameters)")
        proba = np.asarray(
            model.predict(X, batch_size=args.batch_size, verbose=0), dtype=np.float32)
        if proba.shape != (n, NUM_CLASSES):
            failures.append(f"{key}: output shape {proba.shape}, expected {(n, NUM_CLASSES)}")
            continue

        frames[key] = make_frame(index, proba)
        arrays[key] = proba

        if args.verify:
            stored_path = os.path.join(args.models, spec["stored_predictions"])
            if not os.path.exists(stored_path):
                print(f"  [verify] no stored vector at {stored_path}")
                continue
            stored = np.load(stored_path)
            if len(stored) != n:
                failures.append(f"{key}: stored vector has {len(stored)} rows, partition has {n}")
                continue
            agree = int((stored == frames[key]["y_pred"].to_numpy()).sum())
            status = "ok" if agree == n else "MISMATCH"
            print(f"  [verify] {agree:,}/{n:,} predictions match the stored vector [{status}]")
            if agree != n:
                failures.append(f"{key}: {n - agree} predictions differ from the stored vector")

    # ---- baselines -------------------------------------------------------
    # Persistence: the next beat keeps the class of the last observed beat.
    frames["persistence"] = make_frame(index, one_hot(y_prev))
    # Majority class in the training distribution: always N.
    frames["majority"] = make_frame(index, one_hot(np.zeros(n, dtype=int)))
    print("persistence baseline: y_pred = y_prev")
    print("majority-class baseline: y_pred = N (class 0)")

    # ---- write -----------------------------------------------------------
    print()
    for key, frame in frames.items():
        path = os.path.join(args.out, f"{key}_protocolB_predictions.csv")
        frame.to_csv(path, index=False)
        print(f"wrote {path}  ({len(frame):,} rows x {frame.shape[1]} columns)")

    geometry_path = os.path.join(args.out, "protocolB_sequence_geometry.csv")
    geometry.to_csv(geometry_path, index=False)
    print(f"wrote {geometry_path}  "
          f"({len(geometry):,} rows x {geometry.shape[1]} columns)")

    if args.npz:
        path = os.path.join(args.out, "protocolB_predictions.npz")
        bundle = {
            "row_index": index["row_index"].to_numpy(),
            "sequence_id": index["sequence_id"].to_numpy().astype("U16"),
            "record_id": record_ids,
            "window_start": window_starts,
            "y_true": y_true,
            "y_prev": y_prev,
            "class_names": np.asarray(CLASS_NAMES, dtype="U8"),
            "rpeak_input": rpeaks_in,
            "rpeak_target": rpeaks_target,
            "fs_hz": fs,
            "target_overlap_samples": geometry["target_overlap_samples"].to_numpy(),
        }
        for key, frame in frames.items():
            bundle[f"{key}_y_pred"] = frame["y_pred"].to_numpy()
            bundle[f"{key}_y_pred_proba"] = frame[PROBA_COLUMNS].to_numpy(dtype=np.float32)
        np.savez_compressed(path, **bundle)
        print(f"wrote {path}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print("  -", failure)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
