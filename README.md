# Next-beat arrhythmia prediction on MIT-BIH — reviewer materials

Code, trained checkpoints, per-sequence predictions and verification artifacts
for *Next-beat arrhythmia prediction from short beat sequences*.

**The task.** Given three consecutive heartbeats B(t−2), B(t−1), B(t), predict
the rhythm class of the **not-yet-observed** beat B(t+1). This is not
retrospective beat classification: the target beat is never part of the input.
That distinction is easy to implement incorrectly, so it is asserted in code —
for a record with beat labels `L`, the window starting at index `i` uses beats
`i, i+1, i+2` and takes target `L[i+3]`.

---

## Verify the central claims without training anything

Every headline number can be checked from the artifacts in this repository in
about five minutes. No GPU, no retraining.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download MIT-BIH from PhysioNet (not redistributed here)
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
# ...or see https://physionet.org/content/mitdb/1.0.0/

cd scripts

# 1. Rebuild the test partition from the raw WFDB files using an INDEPENDENT
#    reimplementation of the preprocessing, run the shipped checkpoints over
#    it, and check the result against the stored prediction vectors from the
#    original training runs. Exits non-zero on any disagreement.
python export_protocolB_predictions.py --data <mitdb> \
    --models ../checkpoints --out ../results --verify

# 2. Per-record bootstrap intervals, from the exported predictions.
python record_level_bootstrap.py --exports ../results --out ../results

# 3. Beat-window overlap / leakage audit, from the raw annotations.
python window_overlap_check.py --data <mitdb> --out ../results

# 4. Structural smoke test (8 assertions, ~2 min).
python smoke_test.py --data <mitdb>
```

Step 1 reports `6,103/6,103 predictions match the stored vector [ok]` for both
models; the recomputed probability vectors agree with the stored ones to
2.3e-09 (Model-I) and 1.8e-07 (Model-II), i.e. float32 rounding.

---

## What a reviewer will most likely want to check

| Claim | Where |
|---|---|
| Inter-patient partition is 6,103 sequences, record 107 contributes 0 | `scripts/export_protocolB_predictions.py --verify` |
| Target beat is outside the input window | `scripts/smoke_test.py`, assertion 4 |
| Macro-averages use all 8 classes, not just those present | `ecgnextbeat/metrics.py`; `docs/VERIFICATION.md` §1 |
| Class index 4 = AESC, 5 = ABERR | `ecgnextbeat/config.py`; `docs/VERIFICATION.md` §2 |
| Model-I has 1,577,832 parameters, not ~3.6M | `scripts/smoke_test.py`, assertion 5 |
| Persistence baseline beats both models | `results/persistence_protocolB_predictions.csv` |
| Published architecture fails the same way | `docs/external_baseline_note.md` |
| Beat windows overlap only in premature classes, never at QRS | `results/window_overlap_by_class.csv` |

---

## Layout

```
ecgnextbeat/     the pipeline: config, data, models, metrics, train
scripts/         one entry point per result (see docs/PIPELINE.md)
checkpoints/     trained Protocol B models + stored prediction vectors
results/         per-sequence predictions and derived tables
docs/            VERIFICATION.md, FINDINGS.md, PIPELINE.md
```

`results/*_protocolB_predictions.csv` all share one row order and one
`sequence_id`, so they join directly: `row_index`, `sequence_id`, `record_id`,
`y_true`, `y_prev` (last observed beat — the transition/non-transition split),
`y_pred`, `y_pred_conf`, and the full 8-class probability vector.
`results/protocolB_sequence_geometry.csv` joins on the same key and carries
R-peak positions, RR intervals and window-overlap measures.

- **`docs/VERIFICATION.md`** — what reproduced, what did not, and five
  discrepancies found and corrected between the original runs and the revision.
- **`docs/FINDINGS.md`** — the five follow-up experiments, including what was
  deliberately not run and why.
- **`docs/PIPELINE.md`** — full reproduction instructions, protocols, seeds and
  measured per-epoch costs.

## Protocols

**Protocol A (intra-patient).** All records pooled, then sequences split 80/20
and 90/10. Sequences from one subject appear on both sides. Reported for
comparability with the literature, not as evidence of generalisation.

**Protocol B (inter-patient, record-wise).** Records partitioned *before any
sequence is built*: 35 train, 7 validation, 6 test, asserted disjoint. Test
records 102, 104, 107, 114, 118, 201 give **6,103 sequences** from five records
(107 contains only paced beats and contributes none).

## Data

The MIT-BIH Arrhythmia Database is **not** redistributed here. Download from
PhysioNet: https://physionet.org/content/mitdb/1.0.0/ — cite Moody & Mark (2001)
and Goldberger et al. (2000).

## Reproducibility caveat

`train.set_seed` seeds Python, NumPy and TensorFlow, but CPU training is not
bit-for-bit deterministic — multi-threaded kernel reductions vary between runs.
Expect a few tenths of a percent of drift in accuracy at a fixed seed. The
shipped checkpoints and prediction vectors are the exact artifacts behind the
reported numbers, which is why verification is framed around them rather than
around retraining.
