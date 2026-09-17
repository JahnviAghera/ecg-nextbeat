# Additional experiments — findings

Companion to `VERIFICATION.md`. Covers the five deliverables requested after the
first revision. Every file referenced is in `revision/exports/`; all code is in
`ecg-nextbeat/`.

Status at time of writing: deliverables 3, 4 and 5 are complete. Deliverable 2
is running. Deliverable 1 was deliberately not run — see below.

---

## 1. Multi-seed replication — NOT RUN (deferred by decision)

**Files:** `ecg-nextbeat/scripts/run_multiseed.py` (runnable, untested against a
full sweep).

The sweep is 5 seeds × 2 protocols × 2 models = 20 training runs. Measured on
this machine (10-core Apple Silicon, no GPU, TensorFlow 2.21): Model-I takes 38
s/epoch and Model-II 223 s/epoch on the 74,163-sequence Protocol B training
split, and historical runs stopped anywhere between 16 and 91 epochs. At a
representative 60 epochs that is ~38 min per Model-I run and ~3.7 h per
Model-II run, so the full sweep is **~43 CPU-hours**. It was deferred in favour
of deliverable 2. The script is resumable — it appends to
`multiseed_results.csv` after each run and skips rows already present — so the
sweep can be run in stages or moved to a GPU unchanged.

**What it would settle.** The manuscript's headline numbers are single runs at
seed 42. Model-I and Model-II differ by 0.4 percentage points of accuracy
(0.5909 vs 0.5869) and McNemar already says that gap is not significant
(χ² = 0.882, p = 0.348). A seed sweep would establish whether the *macro-F1*
ordering is similarly unstable, which matters more, because macro-F1 is where
the two models differ most (0.1406 vs 0.1058 under the original convention) and
that gap currently rests on one run each.

---

## 2. Matched-parameter-budget ablation — DESIGN COMPLETE, TRAINING IN PROGRESS

**Files:** `matched_ablation_budgets.csv` (complete),
`matched_ablation_results.csv` and `matched_ablation_summary.csv` (written
incrementally as runs complete).

The published ablation (Table 6) compares four variants whose sizes span a
factor of 4.4 — CNN-only 836,808 parameters against CNN+BiLSTM+Transformer
3,667,656, a **338% spread**. Any difference between those variants confounds
the temporal mechanism with raw capacity, which is the objection this
deliverable answers. Each variant's non-convolutional widths (temporal block and
dense head) were rescaled by a bisection search onto a common 1,800,000-parameter
budget, chosen to match the CNN+BiLSTM variant in the original ablation. The
convolutional front end is held fixed at (64, 128, 256) filters across all four,
so the comparison isolates the temporal block rather than swapping feature
extractors as well:

| Variant | Original | Width × | Matched | From target |
|---|---|---|---|---|
| CNN_only | 836,808 | 2.5013 | 1,800,397 | +0.02% |
| CNN_BiLSTM | 1,823,944 | 0.8722 | 1,800,177 | +0.01% |
| CNN_Transformer | 1,694,920 | 1.5041 | 1,799,779 | −0.01% |
| CNN_BiLSTM_Transformer | 3,667,656 | 0.5846 | 1,803,078 | +0.17% |

Spread across the four is now **0.18%**, against 338% originally — comfortably
inside the ±10% the design called for. Three seeds (42, 52, 62) per variant.
Findings will be appended when the runs finish.

---

## 3. Beat-window overlap check — COMPLETE

**Files:** `window_overlap_by_class.csv`, `window_overlap_by_class_detail.csv`,
`protocolB_sequence_geometry.csv`.

Each beat is a 180-sample window centred on its R-peak, spanning [R−90, R+90),
so beat *i*'s window runs into beat *i+1*'s whenever the RR interval is under
180 samples — 0.5 s at 360 Hz, a rate above 120 bpm. Corpus-wide, **2.60% of
beats (2,430 of 93,340) overlap their successor**, and 2.93% within the Protocol
B test partition. The aggregate is unalarming; the breakdown is not. Overlap is
concentrated almost entirely in the premature and ectopic classes — NPC 30.1%,
ABERR 20.7%, APC 14.7% — against N 2.6% and **exactly 0% for LBBB and RBBB**.
This is mechanically unsurprising, since premature beats are defined by short
coupling intervals, but it means window contamination is worst in precisely the
minority classes that macro-averaged metrics weight most heavily. At sequence
level, 179 of the 6,103 test sequences (2.93%) have a last-input-beat window
that reaches into the target beat's window, rising to **39.2% of ABERR targets**
and 12.5% of APC targets.

**The important negative result:** the input window never reaches the target
R-peak in any of the 6,103 test sequences. The minimum prev-to-target RR
interval is 121 samples against the 90 that would be required, and where overlap
occurs it averages 21.9 samples — 12.2% of the target window, confined to the
pre-QRS baseline and P-wave region. So the target beat's QRS complex is never
observed by the model, and the next-beat formulation survives the audit. The
honest statement for the manuscript is that a small, class-dependent fraction of
sequences see the leading edge of the target beat's window but never its QRS,
with the per-class percentages disclosed. As a sanity check on whether this
matters: accuracy on the 179 overlapping sequences is *higher* than on the
5,924 clean ones for both models (0.693 vs 0.588), but so is the class mix —
and the persistence baseline moves the opposite way (0.816 vs 0.934), which is
what you would expect if overlapping sequences are simply the ones where the
rhythm is changing.

---

## 4. External baseline — COMPLETE

**Files:** `external_baseline_protocolB_predictions.csv`,
`external_baseline_per_class.csv`, `external_baseline_note.md`,
`external_baseline_metrics.json`.

Architecture: the residual 1-D CNN of **Kachuee, Fazeli & Sarrafzadeh (2018)**,
*ECG Heartbeat Classification: A Deep Transferable Representation* (ICHI 2018).
It was chosen over the other candidates because it is the most completely
specified — the paper gives the full layer table (initial Conv1D(32,5), five
residual blocks of two Conv1D(32,5) with skip connection and max-pooling(5,
stride 2), then two dense layers) and uses no proprietary or undisclosed
preprocessing, so reimplementing it does not require guessing at a front end.
The only changes are those the task demands: the backbone is wrapped in
TimeDistributed so it encodes all three input beats, and the softmax predicts
beat *i+1* rather than a beat inside the window. Filter counts, kernel size,
depth, pooling and the dense head are unchanged; beat windows are 180 samples
rather than 187 to match this paper's preprocessing.

Retrained from scratch on Protocol B (seed 42, 61,256 parameters, 16 epochs,
5.2 min) and evaluated on the identical 6,103-sequence partition, it reaches
**accuracy 0.5877, balanced accuracy 0.1618, macro-F1 0.0926, weighted F1
0.4483, MCC −0.0516, κ −0.0170**. That places it between Model-I (0.5909) and
Model-II (0.5869), **below the majority-class baseline (0.6054)** and far below
persistence (0.9300). Per-class, it collapses almost entirely onto N —
sensitivity 0.971 for N and **0.000 for every other class**, including the 2,162
RBBB sequences.

**This is the most useful result of the five for the manuscript.** It converts
the paper's central claim from an admission about two particular models into a
property of the task: an independently published architecture, given the same
data, protocol and target, fails in the same way and in the same direction. It
directly answers the "not attempted" entry in `VERIFICATION.md` and removes the
reading that the poor Protocol B numbers reflect a defect in the authors' two
architectures. It should be stated as a lower bound on the published model — it
was designed for retrospective single-beat classification and is being asked to
do something else — not as a refutation of Kachuee et al.

---

## 5. Public repository — COMPLETE

**Files:** `ecg-nextbeat/` (package, 8 scripts, README, pinned requirements).

`ecgnextbeat/` holds the pipeline — `config.py` (protocol constants, record
splits, class map, seeds), `data.py` (beat extraction, sequence construction,
both protocols' split logic), `models.py` (Model-I, Model-II, the four ablation
variants width-parameterised, the external baseline), `metrics.py`, `train.py`.
`scripts/` holds one entry point per result: `reproduce_tables.py` for Tables 3,
5, 6, 7 and 8, the three experiment drivers, and the three analysis scripts that
need no training. The README pins exact versions (Python 3.11.15, TensorFlow
2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, wfdb 4.3.1),
links to PhysioNet instead of shipping the data, documents the seed behind every
published number, and reports measured per-epoch costs so a reader can budget a
sweep before starting one.

Two properties worth noting. First, `export_protocolB_predictions.py --verify`
rebuilds the test partition from the raw WFDB files using an independent
reimplementation of the preprocessing and checks the result against the stored
prediction vectors from the original training runs: all 6,103 predictions match
for both models, and the recomputed probability vectors agree to 2.3e-09
(Model-I) and 1.8e-07 (Model-II). Second, `smoke_test.py` asserts the properties
that are easy to break silently — in particular that the target is
`labels[i+3]` and never inside the input window, and that macro-averages use all
eight classes as the denominator.

**Clean-environment verification.** A fresh virtualenv was created from
`requirements.txt` alone, outside the project tree. pip resolved to the exact
pinned versions (TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.5,
scikit-learn 1.9.0, wfdb 4.3.1, SciPy 1.17.1); all five script entry points
parsed their arguments; and `smoke_test.py` passed **8/8** checks in that
environment — imports, record-split disjointness (35/7/6), beat extraction,
sequence construction with the target asserted outside the input window, all
three model builders matching their published parameter counts, the matched
budgets landing within 0.18%, a two-epoch train/predict cycle with normalised
probability vectors, and the eight-class macro-average denominator. What was
*not* done in the clean environment is a full training reproduction of any
published table; that is a matter of CPU-hours, not of packaging.

**Caveat on reproducibility.** `train.set_seed` seeds Python, NumPy and
TensorFlow, but CPU training is not bit-for-bit deterministic: multi-threaded
kernel reductions vary between runs. Expect differences of a few tenths of a
percent in accuracy at a fixed seed. This is documented in the README and is an
argument for reporting the multi-seed standard deviation rather than a single
run.
