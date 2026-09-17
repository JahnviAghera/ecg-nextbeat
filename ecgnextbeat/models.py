"""Model architectures: Model-I, Model-II, the four ablation variants, and the
matched-parameter-budget versions of those variants."""

from __future__ import annotations

from tensorflow.keras import layers, models

from .config import BEAT_LENGTH, NUM_CLASSES, SEQUENCE_LENGTH


# --------------------------------------------------------------------------
# Shared blocks
# --------------------------------------------------------------------------
def conv_stage(x, filters: int, dropout: float = 0.3):
    x = layers.TimeDistributed(layers.Conv1D(filters, 5, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.Conv1D(filters, 5, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    return x


def transformer_block(x, num_heads: int = 8, key_dim: int = 64,
                      ff_dim: int = 512, dropout: float = 0.3):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(x + ffn)


def _head(x, units, dropout: float = 0.3):
    for size in units:
        x = layers.Dense(size, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax")(x)


def _inputs(sequence_length: int = SEQUENCE_LENGTH):
    return layers.Input(shape=(sequence_length, BEAT_LENGTH, 1))


# --------------------------------------------------------------------------
# Headline models, exactly as trained for the manuscript
# --------------------------------------------------------------------------
def build_model1(sequence_length: int = SEQUENCE_LENGTH):
    """Model-I: TimeDistributed CNN + LSTM. 1,577,832 parameters at L=3."""
    inputs = _inputs(sequence_length)
    x = layers.TimeDistributed(layers.Conv1D(32, 5, activation="relu", padding="same"))(inputs)
    x = layers.TimeDistributed(layers.Conv1D(32, 5, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)

    x = layers.TimeDistributed(layers.Conv1D(64, 5, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 5, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.25))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(128, activation="tanh", return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Model_I_TD_CNN_LSTM")


def build_model2(sequence_length: int = SEQUENCE_LENGTH):
    """Model-II: CNN + BiLSTM + Transformer. 3,667,656 parameters at L=3."""
    return build_variant("CNN_BiLSTM_Transformer", width=1.0,
                         name="Model_II_CNN_BiLSTM_Transformer",
                         sequence_length=sequence_length)


# --------------------------------------------------------------------------
# Ablation variants, width-parameterised
# --------------------------------------------------------------------------
VARIANTS = ("CNN_only", "CNN_BiLSTM", "CNN_Transformer", "CNN_BiLSTM_Transformer")


def build_variant(variant: str, width: float = 1.0, name: str | None = None,
                  sequence_length: int = SEQUENCE_LENGTH):
    """One of the four ablation variants, with every non-convolutional width
    scaled by `width`.

    The convolutional feature extractor is held fixed at (64, 128, 256) across
    all four variants and all widths. Only the temporal block and the dense head
    scale, so a matched-budget comparison isolates the temporal mechanism rather
    than confounding it with a different front end.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    def scaled(base: int) -> int:
        return max(8, int(round(base * width)))

    lstm_units = scaled(256)
    ff_dim = scaled(512)
    head_units = (scaled(512), scaled(256))

    inputs = _inputs(sequence_length)
    x = conv_stage(inputs, 64)
    x = conv_stage(x, 128)
    x = conv_stage(x, 256)
    x = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)

    if "BiLSTM" in variant:
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
        x = layers.BatchNormalization()(x)

    if "Transformer" in variant:
        key_dim = max(8, x.shape[-1] // 8)
        x = transformer_block(x, num_heads=8, key_dim=key_dim, ff_dim=ff_dim)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = _head(x, head_units)
    return models.Model(inputs, outputs, name=name or f"Ablation_{variant}")


def fit_to_budget(variant: str, target: int, tolerance: float = 0.10,
                  low: float = 0.05, high: float = 6.0, steps: int = 40):
    """Find a width multiplier putting `variant` within `tolerance` of `target`.

    Bisects on the width scalar, which is monotone in parameter count. Returns
    (width, parameter_count). Raises if the budget is unreachable — which
    happens when even width -> 0 leaves the fixed convolutional stack above
    target.
    """
    import tensorflow as tf

    def count(width: float) -> int:
        tf.keras.backend.clear_session()
        return build_variant(variant, width).count_params()

    floor = count(low)
    if floor > target * (1 + tolerance):
        raise ValueError(
            f"{variant}: minimum reachable size {floor:,} exceeds "
            f"{target:,} +{tolerance:.0%}; the fixed conv stack is too large")

    # Bisect the full budget of steps and keep the closest match rather than
    # returning on the first width inside tolerance -- an early return can stop
    # 9% off target when a width within 1% exists a few steps further in.
    best = (low, floor)
    for _ in range(steps):
        mid = (low + high) / 2
        params = count(mid)
        if abs(params - target) < abs(best[1] - target):
            best = (mid, params)
        if params == target:
            break
        if params < target:
            low = mid
        else:
            high = mid

    width, params = best
    if abs(params - target) > tolerance * target:
        raise ValueError(f"{variant}: converged to {params:,}, outside "
                         f"{target:,} +/-{tolerance:.0%}")
    return width, params


# --------------------------------------------------------------------------
# External baseline
# --------------------------------------------------------------------------
def build_external_baseline(sequence_length: int = SEQUENCE_LENGTH):
    """Kachuee et al. (2018), "ECG Heartbeat Classification: A Deep
    Transferable Representation" — the residual 1-D CNN, adapted to the
    next-beat task.

    The original operates on a single 187-sample beat: an initial Conv1D(32, 5)
    followed by five residual blocks, each two Conv1D(32, 5) with a skip
    connection and max-pooling, then two dense layers. The only changes here are
    those the task demands: the backbone is wrapped in TimeDistributed so it
    sees all three input beats, the per-beat representations are concatenated,
    and the softmax predicts the class of the NEXT beat rather than the beat in
    the window. Filter widths, depth, kernel size and the dense head are
    unchanged from the paper.
    """
    def residual_block(x, filters: int = 32):
        skip = x
        y = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        y = layers.Conv1D(filters, 5, padding="same")(y)
        y = layers.Add()([y, skip])
        y = layers.Activation("relu")(y)
        return layers.MaxPooling1D(5, strides=2, padding="same")(y)

    beat_in = layers.Input(shape=(BEAT_LENGTH, 1))
    h = layers.Conv1D(32, 5, padding="same")(beat_in)
    for _ in range(5):
        h = residual_block(h, 32)
    h = layers.Flatten()(h)
    h = layers.Dense(32, activation="relu")(h)
    backbone = models.Model(beat_in, h, name="Kachuee2018_backbone")

    inputs = _inputs(sequence_length)
    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="External_Kachuee2018_NextBeat")


BUILDERS = {
    "model1": build_model1,
    "model2": build_model2,
    "external": build_external_baseline,
}
