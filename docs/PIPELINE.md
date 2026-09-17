# Next-beat arrhythmia prediction on MIT-BIH

Code for *Next-beat arrhythmia prediction from short beat sequences*: given the
three preceding heartbeats B(t−2), B(t−1), B(t), predict the rhythm class of the
**not-yet-observed** beat B(t+1).

This is not retrospective beat classification. The target beat is never part of
the input. That distinction is easy to implement incorrectly, so it is asserted
in code: for a record with beat labels `L`, the window starting at index `i`
uses beats `i, i+1, i+2` and is assigned target `L[i+3]`.

## What is here

```
ecgnextbeat/
  config.py     protocol constants, record splits, class map, seeds
  data.py       beat extraction, sequence construction, Protocol A/B splits
  models.py     Model-I, Model-II, ablation variants, external baseline
  metrics.py    the metric suite (macro-averages over all eight classes)
  train.py      training loop with explicit seed control
scripts/
  reproduce_tables.py         Tables 3, 5, 6, 7, 8
  run_multiseed.py            multi-seed replication of the headline numbers
  run_matched_ablation.py     parameter-matched architecture ablation
  run_external_baseline.py    Kachuee et al. (2018) retrained on this task
  export_protocolB_predictions.py   per-sequence predictions + beat geometry
  window_overlap_check.py     beat-window overlap / leakage audit
  record_level_bootstrap.py   per-record bootstrap intervals
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact versions used for every number in the paper are pinned in
`requirements.txt`: Python 3.11.15, TensorFlow 2.21.0, Keras 3.15.1,
NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, wfdb 4.3.1.

## Data

The raw MIT-BIH Arrhythmia Database is **not** included. Download it from
PhysioNet:

- https://physionet.org/content/mitdb/1.0.0/
- `wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/`

Point `--data` at the directory holding the `.hea`, `.atr` and `.dat` files.
All 48 records are used. Record 107's `.dat` is optional — it contains only
paced beats, none of which map to the eight retained classes, so it contributes
no sequences either way.

## Protocols

**Protocol A (intra-patient).** Every record is pooled, then sequences are split
80/20 into train+val and test, and the train+val pool is split 90/10. Sequences
from the same subject appear on both sides. Reported for comparability with the
published literature, not as a measure of generalisation.

**Protocol B (inter-patient, record-wise).** Records are partitioned *before any
sequence is constructed*: 35 train, 7 validation, 6 test. No window spans a
record boundary and the three record sets are asserted disjoint. Test records
are 102, 104, 107, 114, 118, 201, giving **6,103 test sequences** from five
records.

Record lists are in `ecgnextbeat/config.py` and match Appendix A2.

## Classes

Eight MIT-BIH classes, mapped from annotation symbols:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Name | N | LBBB | RBBB | APC | AESC | ABERR | NPC | NESC |
| Symbol | `N` | `L` | `R` | `A` | `e` | `a` | `J` | `j` |

Index 4 is AESC (`e`) and index 5 is ABERR (`a`). An earlier version of the
analysis had these two reversed; see `VERIFICATION.md` in the revision bundle.

Macro-averages are taken over **all eight** defined classes, including those
with zero support in a partition. Averaging over only the classes present makes
the denominator depend on the predictions.

## Reproducing the tables

All table scripts default to seed 42, the seed behind the headline single-run
results in the paper.

```bash
cd scripts

# Table 3 — per-class F1, Protocol A
python reproduce_tables.py --data ../data --out ../results --table table3

# Table 5 — per-class F1, Protocol B
python reproduce_tables.py --data ../data --out ../results --table table5

# Table 6 — architecture ablation, original (unmatched) sizes
python reproduce_tables.py --data ../data --out ../results --table table6

# Table 7 — sequence-length ablation, 1/2/3/5/10 beats
python reproduce_tables.py --data ../data --out ../results --table table7

# Table 8 — temporal-order sanity check
python reproduce_tables.py --data ../data --out ../results --table table8

# or everything
python reproduce_tables.py --data ../data --out ../results --table all
```

Analyses that need no training, because they read the exported per-sequence
predictions:

```bash
python export_protocolB_predictions.py --data ../data \
    --models ../data/saved_models --out ../results --npz --verify
python window_overlap_check.py --data ../data --out ../results
python record_level_bootstrap.py --exports ../results --out ../results
```

`--verify` checks freshly computed predictions against the stored prediction
vectors from the original training runs and exits non-zero on any disagreement.

## Additional experiments

```bash
python run_multiseed.py --data ../data --out ../results
python run_matched_ablation.py --data ../data --out ../results
python run_external_baseline.py --data ../data --out ../results
```

`run_multiseed.py` and `run_matched_ablation.py` append to their CSV after every
run and skip work already present, so an interrupted sweep resumes where it
stopped.

## Seeds

| Result | Seed(s) |
|---|---|
| Headline single-run numbers (Tables 3–10) | 42 |
| Multi-seed replication | 42, 52, 62, 72, 82 |
| Matched-budget ablation | 42, 52, 62 |
| Bootstrap resampling (2,000 draws) | 42 |
| Stratified-random baseline (100 draws) | 42 |

`train.set_seed` seeds Python, NumPy and TensorFlow. It does **not** make CPU
training bit-for-bit deterministic — multi-threaded reductions in the TensorFlow
kernels vary between runs. Set `TF_DETERMINISTIC_OPS=1` and restrict the thread
pools if exact reproduction matters more than wall-clock time. Expect
run-to-run differences of a few tenths of a percent in accuracy even at a fixed
seed; this is why the multi-seed table reports a standard deviation.

## Compute

Every result in the paper was produced on CPU. Measured on a 10-core Apple
Silicon machine (TensorFlow 2.21, no GPU), Protocol B, 74,163 oversampled
training sequences:

| Model | Parameters | Seconds / epoch |
|---|---|---|
| Model-I (CNN-LSTM) | 1,577,832 | 38 |
| Model-II (CNN-BiLSTM-Transformer) | 3,667,656 | 223 |
| External baseline (Kachuee 2018) | 61,256 | ~25 |

Protocol A costs about the same per epoch: pooling all 48 records and taking
80%/90% of them leaves a training split of comparable size to Protocol B's
35 records after oversampling.

Runs stop on early stopping (patience 15, 100-epoch cap) and in practice have
taken anywhere from 16 to 91 epochs. At 60 epochs that is roughly 38 minutes
for Model-I and 3.7 hours for Model-II per run. **A full 5-seed × 2-protocol ×
2-model sweep is therefore on the order of 40 CPU-hours, and the 4-variant ×
3-seed matched ablation another 30.** Both sweep scripts are resumable, so they
can be run in stages. A CUDA GPU reduces this by roughly an order of magnitude
and is strongly recommended for the sweeps.

## Baselines

Three baselines, all on the identical Protocol B partition:

- **Persistence** — predict `class(B(t+1)) = class(B(t))`. No parameters, no
  signal morphology. The natural null model for a sequential prediction task,
  and the one the trained models must beat to be worth anything.
- **Majority class** — always predict N.
- **Stratified random** — draw from the empirical test prior, mean of 100 draws.

## Licence and citation

Released for review and reuse. The MIT-BIH Arrhythmia Database is distributed by
PhysioNet under its own terms; cite Moody & Mark (2001) and Goldberger et al.
(2000) alongside this work.
