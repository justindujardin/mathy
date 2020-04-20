import itertools
from typing import Any, Optional, Tuple

import numpy
import tensorflow as tf
import tqdm
from fragile.core import HistoryTree, Swarm
from fragile.core.utils import random_state
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


def calculate_contrastive_loss(
    hidden_aa: tf.Tensor,
    hidden_ab: tf.Tensor,
    hidden_ba: tf.Tensor,
    hidden_bb: tf.Tensor,
    hidden_norm: bool = True,
    temperature: float = 1.0,
    weights: float = 1.0,
    writer: tf.summary.SummaryWriter = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute loss for model.
  Args:
    hidden_aa: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_ab: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_ba: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_bb: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    weights: a weighting number or vector.
  Returns:
    A logits tensor for the predictive task
    A labels tensor for the predictive task
    A loss scalar for agreement between hidden_aa and hidden_ab
    A loss scalar for agreement between hidden_ba and hidden_bb
    A loss scalar for disagreement between hidden_aa and hidden_ba
    A loss scalar for disagreement between hidden_ab and hidden_bb
  """
    # Get (normalized) hidden_aa and hidden_ab.
    if hidden_norm:
        hidden_aa = tf.math.l2_normalize(hidden_aa, -1)
        hidden_ab = tf.math.l2_normalize(hidden_ab, -1)
        hidden_ba = tf.math.l2_normalize(hidden_ba, -1)
        hidden_bb = tf.math.l2_normalize(hidden_bb, -1)
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    loss_a = cosine_loss(hidden_aa, hidden_ab)
    loss_b = cosine_loss(hidden_ba, hidden_bb)
    loss_contrast_a = -cosine_loss(hidden_aa, hidden_ba)
    loss_contrast_b = -cosine_loss(hidden_ab, hidden_bb)
    logits_match_one = tf.concat([hidden_aa, hidden_ab], axis=-1)
    logits_match_two = tf.concat([hidden_ba, hidden_bb], axis=-1)
    logits_mismatch_one = tf.concat([hidden_aa, hidden_bb], axis=-1)
    logits_mismatch_two = tf.concat([hidden_ba, hidden_ab], axis=-1)
    batch_size = tf.shape(hidden_aa)[0]
    labels_mismatch = tf.zeros((batch_size, 1))
    labels_match = tf.ones((batch_size, 1))
    logits = tf.concat(
        [logits_match_one, logits_mismatch_one, logits_match_two, logits_mismatch_two],
        axis=0,
    )
    labels = tf.concat(
        [labels_match, labels_mismatch, labels_match, labels_mismatch], axis=0
    )
    return logits, labels, loss_a, loss_contrast_a, loss_b, loss_contrast_b


class ContrastiveMathModel:
    def __new__(
        cls, input_shape: Tuple[int, int, int], vocab_len: int, out_dim: int = 32
    ):
        model = Sequential(
            [
                tf.keras.layers.Embedding(
                    input_shape=input_shape,
                    input_dim=vocab_len + 1,
                    output_dim=64,
                    name="inputs",
                    mask_zero=True,
                ),
                Dense(256, activation="relu", name="hidden"),
                Flatten(),
                Dense(out_dim, name="output"),
            ]
        )
        model.compile(
            loss="mean_squared_error",
            optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
            metrics=["accuracy"],
        )
        model.summary()
        return model


class PredictiveMathModel:
    """Predict if two vectors are different views of the same expression
    or views of different expressions"""

    def __new__(cls, input_shape: Tuple[int, int, int]):
        model = Sequential(
            [
                Dense(
                    128, activation="relu", name="head", batch_input_shape=input_shape
                ),
                Dense(1, name="output"),
            ]
        )
        model.compile(
            loss=tf.nn.softmax_cross_entropy_with_logits,
            optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
            metrics=["accuracy"],
        )
        model.summary()
        return model


class ContrastiveModelTrainer:
    def __init__(
        self,
        vocab_len: int,
        input_shape: tuple,
        writer: tf.summary.SummaryWriter = None,
        model_file: str = None,
        batch_size: int = 12,
    ):
        out_dim = 32
        self.model = ContrastiveMathModel(
            input_shape=input_shape, vocab_len=vocab_len, out_dim=out_dim
        )
        self.predict = PredictiveMathModel(input_shape=(batch_size * 4, out_dim * 2))
        self.model_file = model_file
        self.writer = writer

    def train(
        self, batch_fn: Any, batches: int = 100, epochs: int = 10, verbose: int = 0,
    ):
        for i in range(epochs):
            epoch_loss: float = 0.0
            loss_a_agreement: float = 0.0
            loss_ab_disagreement: float = 0.0
            loss_b_agreement: float = 0.0
            loss_ba_disagreement: float = 0.0
            print(f"Epoch: {i}")
            for j in tqdm.tqdm(range(batches)):
                aa_batch, ab_batch, ba_batch, bb_batch = batch_fn()
                step_losses = self.train_step(aa_batch, ab_batch, ba_batch, bb_batch)
                a_agree, ab_disagree, b_agree, ba_disagree = step_losses
                loss_a_agreement += a_agree
                loss_ab_disagreement += ab_disagree
                loss_b_agreement += b_agree
                loss_ba_disagreement += ba_disagree
                epoch_loss += a_agree + b_agree + ab_disagree + ba_disagree
            if self.writer is not None:
                with self.writer.as_default():
                    step = self.model.optimizer.iterations
                    tf.summary.scalar("loss/a_agreement", loss_a_agreement, step=step)
                    tf.summary.scalar("loss/b_agreement", loss_b_agreement, step=step)
                    tf.summary.scalar(
                        "loss/ab_disagreement", loss_ab_disagreement, step=step
                    )
                    tf.summary.scalar(
                        "loss/ba_disagreement", loss_ba_disagreement, step=step
                    )
                    tf.summary.scalar("loss/total", epoch_loss, step=step)
                    for var in self.model.trainable_variables:
                        tf.summary.histogram(var.name, var, step=step)
                self.model.save(self.model_file)
            print(
                f"total: {epoch_loss} aa: {loss_a_agreement} bb: {loss_b_agreement} dab: {loss_ab_disagreement} dba: {loss_ba_disagreement}"
            )

    def train_step(self, aa, ab, ba, bb):
        with tf.GradientTape(persistent=True) as tape:
            aa = self.model(aa)
            ab = self.model(ab)
            ba = self.model(ba)
            bb = self.model(bb)
            logits, labels, a, ab, b, ba = calculate_contrastive_loss(
                aa, ab, ba, bb, self.writer
            )
            self.predict.train_on_batch(logits, labels)
            total_loss = a + ab + b + ba
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights)
            )
        return a, ab, b, ba


def peek(iterable) -> Optional[Tuple[Any, Any]]:
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def prepare_batch(batch_observations):
    return batch_observations
