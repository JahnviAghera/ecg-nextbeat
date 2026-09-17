# Verification report

Every number in the revised manuscript and the response letter was recomputed before
being written. This file records what was checked, what reproduced, and what did not.

## Method

The inter-patient test-label vector was rebuilt from the raw MIT-BIH `.atr` / `.hea`
files using an independent reimplementation of the preprocessing rules, not loaded from
the training pipeline. Stored per-sequence prediction vectors (`*.npy`) were then scored
against it with a single metric implementation.

Reconstructed partition: **6,103 sequences** from records 102 (96), 104 (160),
114 (1,829), 118 (2,258), 201 (1,760). Record 107 contributes 0 — it contains only
paced beats, none of which map to the eight retained classes.

Class support: N 3,695 · LBBB 0 · RBBB 2,162 · APC 136 · AESC 0 · ABERR 97 · NPC 3 · NESC 10.

## Reproduced exactly

| Configuration | Accuracy | Balanced acc. | Weighted F1 | MCC | Cohen's κ |
|---|---|---|---|---|---|
| Model-I (CNN-LSTM) | 0.590857 | 0.172724 | 0.530241 | 0.146389 | 0.112099 |
| Model-II (CNN-BiLSTM-Transformer) | 0.586924 | 0.161570 | 0.448400 | −0.042865 | −0.015026 |
| Model-I + focal & augmentation | 0.529739 | 0.147008 | 0.419607 | −0.110427 | −0.067321 |
| Model-II + focal & augmentation | 0.487793 | 0.149785 | 0.528833 | 0.316652 | 0.255960 |

All per-class precision, sensitivity, specificity and F1 values also reproduced exactly.
Prediction vector lengths matched the reconstructed label vector for all four
configurations (6,103 each), confirming the training-time preprocessing agrees with the
independent reimplementation.

## Discrepancies found and resolved

**1. Macro-averaging denominator.** Stored macro-precision / recall / F1 for the two
record-wise runs were averaged over the classes appearing in `y_true ∪ y_pred` — a
denominator that varies with the predictions — while the two focal-loss runs used all
eight classes. For Model-I this is the difference between macro-F1 0.1406 (7 classes)
and 0.1230 (8 classes) on identical predictions. The revision fixes the convention at
eight classes throughout and states it in Section 3.6. The architecture-ablation table
reports only convention-invariant metrics (accuracy, balanced accuracy, weighted F1,
MCC, κ) because per-variant predictions were not retained.

**2. Class-index labelling error.** `record_wise/model_II/aesc/aesc_analysis.json` is
labelled `AESC` with `AESC_class_id: 5` and 97 test samples. Index 5 is ABERR
(symbol `a`); AESC is index 4 (symbol `e`). The 97 sequences — all from record 201 —
are ABERR. AESC has 16 sequences in the entire corpus and **zero** in the inter-patient
test partition. Corrected in Section 4.3 and Appendix A4.

**3. Notebook class names.** Cells 48–62 of `ECG_Arrhythmia_Reproduction-4.ipynb` print
`CLASS_NAMES = ["N","S","V","F","Q","Class_5","Class_6","Class_7"]` — AAMI labels applied
to the eight-class MIT-BIH index space. The indices are correct; only the display names
are wrong. Confirmed by matching supports against `Improved_Model_I_per_class.csv`, which
uses the correct names: index 2 = RBBB (2,162), index 5 = ABERR (97). All tables in the
revision use the corrected names.

**4. Model-I parameter count.** The submitted manuscript states ~3.6 M trainable
parameters. Built to the stated specification the model has **1,577,832**
(`model1.count_params()`, confirmed by the layer table: LSTM 1,540,608 + conv 36,192 +
dense 1,032). Corrected throughout.

**5. AESC F1 = 0.0000.** Reported in the submitted Table 2 as a categorical class
failure. The independent Protocol A reproduction gives F1 = 0.50 (Model-I) and 0.02
(Model-II) on the three AESC test sequences. The original value reflects one training
run, not a property of the task.

## New results computed for this revision

**Persistence baseline.** Predicting `class(B(t+1)) = class(B(t))` on the inter-patient
partition gives accuracy **0.9300**, balanced accuracy 0.3763, macro-F1 0.2823,
weighted F1 0.9300, MCC 0.8621, κ 0.8621. Alignment was checked explicitly: for window
*i* the target is `labels[i+3]` and the last observed beat is `labels[i+2]`, giving
vectors of identical length (6,103). The label-persistence rate
`P[class(t+1) = class(t)]` is **0.9300** on this partition.

**Majority-class baseline.** 0.6054 accuracy — above both Model-I (0.5909) and
Model-II (0.5869).

**Stratified-random baseline.** 0.4936 accuracy, mean of 100 draws from the empirical
test prior, seed 42.

## Carried forward from the authors' runs without recomputation

These could not be recomputed because per-variant prediction vectors were not retained.
They are reported as measured by the original runs:

- Architecture ablation (4 variants) — `record_wise/model_II/ablation/ablation_results.csv`
- Sequence-length ablation (5 window lengths) — `sequence_length_ablation.csv`
- Temporal-order sanity check — `temporal_order_sanity_check.csv`
- Conventional single-beat control (11.18% accuracy) — `conventional_vs_nextbeat.csv`
- Inference latency and parameter counts — `model_complexity_inference.csv`
- McNemar test (χ² = 0.882, p = 0.348) — notebook cell 73
- Protocol A reproduction figures — notebook cells 19 and 25

## Not attempted

No external published architecture was re-implemented and retrained under this protocol.
This is stated as a limitation in the manuscript rather than papered over. The persistence
baseline is offered in its place and is exactly reproducible by anyone.
