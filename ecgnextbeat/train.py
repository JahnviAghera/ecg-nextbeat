"""Training loop with explicit seed control."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np

from .config import (BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS,
                     LEARNING_RATE, MIN_LR, REDUCE_LR_FACTOR,
                     REDUCE_LR_PATIENCE)
from .metrics import all_metrics


def set_seed(seed: int) -> None:
    """Seed every generator that affects training.

    Note that this does not make CPU training bit-for-bit deterministic:
    multi-threaded reductions in the TensorFlow kernels remain a source of
    run-to-run variation. Set TF_DETERMINISTIC_OPS=1 and restrict the thread
    pools if exact reproduction matters more than wall-clock time.
    """
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


@dataclass
class RunResult:
    metrics: dict
    y_pred: np.ndarray
    y_proba: np.ndarray
    epochs_trained: int
    parameters: int
    seconds: float
    history: dict


def train_and_evaluate(build_fn, partition, seed: int,
                       epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                       checkpoint_dir: str | None = None,
                       verbose: int = 0) -> RunResult:
    """Train one model on one partition and score it on the test split."""
    import tensorflow as tf
    from tensorflow.keras import callbacks, optimizers

    set_seed(seed)
    model = build_fn()
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=EARLY_STOPPING_PATIENCE,
                                restore_best_weights=True, verbose=verbose),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                    patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR,
                                    verbose=verbose),
    ]
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        callback_list.append(callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=verbose))

    started = time.time()
    history = model.fit(partition.X_train, partition.y_train,
                        validation_data=(partition.X_val, partition.y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callback_list, verbose=verbose)
    elapsed = time.time() - started

    proba = np.asarray(model.predict(partition.X_test, batch_size=256, verbose=0),
                       dtype=np.float32)
    y_pred = proba.argmax(axis=1)

    result = RunResult(
        metrics=all_metrics(partition.y_test, y_pred),
        y_pred=y_pred,
        y_proba=proba,
        epochs_trained=len(history.history["loss"]),
        parameters=int(model.count_params()),
        seconds=elapsed,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
    )
    tf.keras.backend.clear_session()
    return result
