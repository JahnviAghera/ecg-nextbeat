"""Fixed experimental constants. These define the protocols and must not drift."""

from __future__ import annotations

SEQUENCE_LENGTH = 3       # beats per input window (paper Eq. 1)
BEAT_LENGTH = 180         # samples per beat window, 90 either side of the R-peak
SMOOTH_WINDOW = 5         # moving-average smoothing applied to the raw signal
MIN_SAMPLES_TARGET = 500  # per-class oversampling floor, training split only
NUM_CLASSES = 8

# Annotation symbol -> class index.
# Index 4 is AESC (symbol 'e'); index 5 is ABERR (symbol 'a'). Reversing these
# two is the labelling error corrected in the revision (VERIFICATION.md, #2).
CLASS_MAP = {"N": 0, "L": 1, "R": 2, "A": 3, "e": 4, "a": 5, "J": 6, "j": 7}
CLASS_NAMES = ["N", "LBBB", "RBBB", "APC", "AESC", "ABERR", "NPC", "NESC"]

# Protocol B (inter-patient, record-wise). Appendix A2.
TRAIN_RECORDS = ["100", "101", "103", "105", "108", "109", "111", "113", "115",
                 "116", "117", "121", "122", "123", "124", "200", "203", "205",
                 "207", "208", "209", "213", "214", "215", "217", "219", "220",
                 "221", "222", "223", "228", "230", "231", "233", "234"]
VAL_RECORDS = ["106", "112", "119", "202", "210", "212", "232"]
TEST_RECORDS = ["102", "104", "107", "114", "118", "201"]

# Protocol A (intra-patient) pools every record and splits at sequence level.
ALL_RECORDS = TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS

# Seeds used for the headline single-run results already in the manuscript.
HEADLINE_SEED = 42
# Seeds for the multi-seed replication.
REPLICATION_SEEDS = [42, 52, 62, 72, 82]
ABLATION_SEEDS = [42, 52, 62]

# Training hyperparameters, identical for every configuration.
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7
