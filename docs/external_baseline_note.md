# External baseline

Deliverable 4 — external baseline reimplemented and retrained on Protocol B.

Architecture: the residual 1-D CNN of Kachuee, Fazeli & Sarrafzadeh (2018),
"ECG Heartbeat Classification: A Deep Transferable Representation" (ICHI 2018).

Why this one. Of the candidates in the literature review it is the most
completely specified: the paper gives the full layer table (initial Conv1D(32,5),
five residual blocks of two Conv1D(32,5) with a skip connection and
max-pooling(5, stride 2), then two dense layers), uses no proprietary or
undisclosed preprocessing, and its reference implementation has been reproduced
widely enough that the architecture is unambiguous. Reimplementing a model whose
preprocessing is under-described would confound the architecture with a guess at
its front end.

What was changed, and why. The original classifies a single 187-sample beat that
has already occurred. The task here is to predict the class of the beat that
follows a three-beat window. The backbone is therefore wrapped in
TimeDistributed so it encodes each of the three input beats, the three
representations are concatenated, and the softmax predicts beat i+1. Filter
counts, kernel size, depth, pooling and the dense head are unchanged. Beat
windows are 180 samples rather than 187, matching this paper's preprocessing so
the comparison isolates the architecture.

This is a lower bound on what the published model can do, not a refutation of
it: it was designed for retrospective classification and is being asked to do
something else.

## Result (Protocol B, seed 42)

- Parameters: 61,256
- Epochs trained: 16
- Test sequences: 6,103

| Metric | Value |
|---|---|
| accuracy | 0.587744 |
| balanced_accuracy | 0.161795 |
| macro_precision | 0.074804 |
| macro_recall | 0.121346 |
| macro_f1 | 0.092553 |
| weighted_f1 | 0.448284 |
| mcc | -0.051600 |
| kappa | -0.016979 |
