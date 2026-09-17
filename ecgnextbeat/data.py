"""Beat extraction, sequence construction, and the Protocol A / B split logic."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import wfdb

from .config import (ALL_RECORDS, BEAT_LENGTH, CLASS_MAP, MIN_SAMPLES_TARGET,
                     NUM_CLASSES, SEQUENCE_LENGTH, SMOOTH_WINDOW,
                     TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS)


@dataclass
class Partition:
    """One protocol's train / validation / test arrays.

    `X_*` are (n, SEQUENCE_LENGTH, BEAT_LENGTH, 1). `test_record_ids` and
    `test_y_prev` are populated for Protocol B, where sequences keep their
    record provenance and temporal neighbour; under Protocol A the test
    sequences are drawn at random across records, so provenance is retained
    per sequence but the partition is not record-disjoint.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_record_ids: np.ndarray
    test_y_prev: np.ndarray
    test_window_starts: np.ndarray


def smooth(signal: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return np.convolve(signal, np.ones(window) / window, mode="same")


def extract_beats(data_path: str, record: str):
    """Z-normalised beat windows, class indices and R-peak sample positions.

    A beat is retained iff its annotation symbol is in CLASS_MAP and its full
    180-sample window lies inside the signal.
    """
    path = os.path.join(data_path, record)
    signal = smooth(wfdb.rdrecord(path).p_signal[:, 0])
    ann = wfdb.rdann(path, "atr")
    half = BEAT_LENGTH // 2

    beats, labels, peaks = [], [], []
    for peak, symbol in zip(ann.sample, ann.symbol):
        if symbol not in CLASS_MAP:
            continue
        if peak - half < 0 or peak + half > len(signal):
            continue
        beat = signal[peak - half: peak + half]
        beats.append((beat - np.mean(beat)) / (np.std(beat) + 1e-8))
        labels.append(CLASS_MAP[symbol])
        peaks.append(int(peak))

    return (np.asarray(beats, dtype=np.float32),
            np.asarray(labels, dtype=int),
            np.asarray(peaks, dtype=int))


def build_sequences(beats: np.ndarray, labels: np.ndarray,
                    sequence_length: int = SEQUENCE_LENGTH):
    """Window i covers beats i .. i+L-1; its target is beat i+L.

    With the default L=3 that is "beats i, i+1, i+2 predict beat i+3".
    Returns (X, y, y_prev, window_start). `y_prev` is the label of the last
    beat actually observed, which the persistence baseline predicts with.
    """
    n = len(labels) - sequence_length
    if n <= 0:
        empty = np.empty((0, sequence_length, BEAT_LENGTH), dtype=np.float32)
        return empty, np.empty(0, int), np.empty(0, int), np.empty(0, int)

    X = np.stack([beats[i: i + sequence_length] for i in range(n)])
    return (X, labels[sequence_length:], labels[sequence_length - 1: -1],
            np.arange(n, dtype=int))


def process_records(data_path: str, records,
                    sequence_length: int = SEQUENCE_LENGTH):
    """Sequences from a list of records, concatenated in the order given."""
    X, y, y_prev, record_ids, starts = [], [], [], [], []
    for record in records:
        beats, labels, _ = extract_beats(data_path, record)
        Xr, yr, pr, sr = build_sequences(beats, labels, sequence_length)
        if len(Xr) == 0:
            continue
        X.append(Xr)
        y.append(yr)
        y_prev.append(pr)
        record_ids.append(np.full(len(yr), int(record), dtype=int))
        starts.append(sr)
    if not X:
        raise RuntimeError("No sequences generated — check the data path.")
    return (np.concatenate(X), np.concatenate(y), np.concatenate(y_prev),
            np.concatenate(record_ids), np.concatenate(starts))


def oversample(X: np.ndarray, y: np.ndarray, seed: int,
               floor: int = MIN_SAMPLES_TARGET):
    """Raise every present class to at least `floor` samples, with replacement.

    Applied to the TRAINING split only — never to validation or test.
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for class_id in range(NUM_CLASSES):
        indices = np.where(y == class_id)[0]
        if len(indices) == 0:
            continue
        selected = rng.choice(indices, size=max(len(indices), floor), replace=True)
        X_parts.append(X[selected])
        y_parts.append(y[selected])

    X_new = np.concatenate(X_parts)
    y_new = np.concatenate(y_parts)
    order = rng.permutation(len(y_new))
    return X_new[order], y_new[order]


def _as_input(X: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    return X.reshape(-1, sequence_length, BEAT_LENGTH, 1)


def build_protocol_b(data_path: str, seed: int,
                     sequence_length: int = SEQUENCE_LENGTH) -> Partition:
    """Records are partitioned BEFORE any sequence is constructed.

    No window spans a record boundary and the three record sets are disjoint,
    so no subject appears in more than one split.
    """
    X_tr, y_tr, _, _, _ = process_records(data_path, TRAIN_RECORDS, sequence_length)
    X_va, y_va, _, _, _ = process_records(data_path, VAL_RECORDS, sequence_length)
    X_te, y_te, prev_te, rec_te, start_te = process_records(
        data_path, TEST_RECORDS, sequence_length)

    assert set(TRAIN_RECORDS).isdisjoint(TEST_RECORDS)
    assert set(VAL_RECORDS).isdisjoint(TEST_RECORDS)

    X_tr, y_tr = oversample(X_tr, y_tr, seed)
    return Partition(_as_input(X_tr, sequence_length), y_tr,
                     _as_input(X_va, sequence_length), y_va,
                     _as_input(X_te, sequence_length), y_te,
                     rec_te, prev_te, start_te)


def _duplicate_singletons(X, y, y_prev, record_ids, starts):
    """Stratified splitting errors on classes with a single member."""
    counts = Counter(y)
    rare = [c for c, n in counts.items() if n < 2]
    if not rare:
        return X, y, y_prev, record_ids, starts
    mask = np.isin(y, rare)
    return (np.concatenate([X, X[mask]]), np.concatenate([y, y[mask]]),
            np.concatenate([y_prev, y_prev[mask]]),
            np.concatenate([record_ids, record_ids[mask]]),
            np.concatenate([starts, starts[mask]]))


def build_protocol_a(data_path: str, seed: int,
                     sequence_length: int = SEQUENCE_LENGTH) -> Partition:
    """Every record is pooled, then sequences are split 80/20 then 90/10.

    Sequences from the same subject appear in both training and test, which is
    why Protocol A accuracy is not comparable to Protocol B accuracy.
    """
    from sklearn.model_selection import train_test_split

    X, y, y_prev, record_ids, starts = process_records(
        data_path, ALL_RECORDS, sequence_length)
    X, y, y_prev, record_ids, starts = _duplicate_singletons(
        X, y, y_prev, record_ids, starts)

    idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y)

    y_tv = y[trainval_idx]
    counts = Counter(y_tv)
    stratify = y_tv if min(counts.values()) >= 2 else None
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.1, random_state=seed, stratify=stratify)

    X_tr, y_tr = oversample(X[train_idx], y[train_idx], seed)
    return Partition(_as_input(X_tr, sequence_length), y_tr,
                     _as_input(X[val_idx], sequence_length), y[val_idx],
                     _as_input(X[test_idx], sequence_length), y[test_idx],
                     record_ids[test_idx], y_prev[test_idx], starts[test_idx])


def build_partition(protocol: str, data_path: str, seed: int,
                    sequence_length: int = SEQUENCE_LENGTH) -> Partition:
    if protocol.upper() in ("B", "PROTOCOL_B"):
        return build_protocol_b(data_path, seed, sequence_length)
    if protocol.upper() in ("A", "PROTOCOL_A"):
        return build_protocol_a(data_path, seed, sequence_length)
    raise ValueError(f"unknown protocol: {protocol}")
