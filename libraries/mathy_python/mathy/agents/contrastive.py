import itertools
from typing import Any, Optional, Tuple

import numpy
import tensorflow as tf
from fragile.core import HistoryTree, Swarm
from fragile.core.utils import random_state
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


def calculate_contrastive_loss(
    hidden1: tf.Tensor,
    hidden2: tf.Tensor,
    hidden3: tf.Tensor,
    hidden4: tf.Tensor,
    hidden_norm: bool = True,
    temperature: float = 1.0,
    weights: float = 1.0,
    writer: tf.summary.SummaryWriter = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute loss for model.
  Args:
    hidden1: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden2: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden3: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden4: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    weights: a weighting number or vector.
  Returns:
    A loss scalar for agreement between hidden1 and hidden2
    A loss scalar for agreement between hidden3 and hidden4
    A loss scalar for disagreement between hidden1 and hidden3
    A loss scalar for disagreement between hidden2 and hidden4
  """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden1 = tf.math.l2_normalize(hidden1, -1)
        hidden2 = tf.math.l2_normalize(hidden2, -1)
        hidden3 = tf.math.l2_normalize(hidden3, -1)
        hidden4 = tf.math.l2_normalize(hidden4, -1)
    batch_size = tf.shape(hidden1)[0]
    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        hidden1, hidden2, weights=weights
    )
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        hidden3, hidden4, weights=weights
    )
    loss_contrast_a = -tf.compat.v1.losses.softmax_cross_entropy(
        hidden1, hidden3, weights=weights
    )
    loss_contrast_b = -tf.compat.v1.losses.softmax_cross_entropy(
        hidden2, hidden4, weights=weights
    )
    return loss_a, loss_contrast_a, loss_b, loss_contrast_b


class ContrastiveMathModel:
    def __new__(cls, input_shape: Tuple[int, int, int], out_dim: int = 32):
        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=input_shape),
                Dense(64, activation="relu"),
                Dense(out_dim),
            ]
        )
        model.compile(
            loss="mean_squared_error",
            optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
            metrics=["accuracy"],
        )
        model.summary()
        return model


class ContrastiveModelTrainer:
    def __init__(
        self, input_shape: tuple, writer: tf.summary.SummaryWriter = None,
    ):
        self.model = ContrastiveMathModel(input_shape)
        self.writer = writer

    def train(
        self,
        compare: Swarm,
        contrast: Swarm,
        batch_size: int = 256,
        epochs: int = 10,
        verbose: int = 0,
    ):
        for i in range(epochs):
            epoch_loss: float = 0.0
            loss_a_agreement: float = 0.0
            loss_ab_disagreement: float = 0.0
            loss_b_agreement: float = 0.0
            loss_ba_disagreement: float = 0.0
            print(f"Epoch: {i}")
            # Generate random sample batches from the compare/contrast swarms
            random_compare_batches = list(
                compare.tree.iterate_nodes_at_random(
                    batch_size=batch_size, names=["observs", "next_observs"]
                )
            )
            assert len(random_compare_batches) > 0, "compare swarm contains no batches"
            random_contrast_batches = list(
                contrast.tree.iterate_nodes_at_random(
                    batch_size=batch_size, names=["observs", "next_observs"]
                )
            )
            assert (
                len(random_contrast_batches) > 0
            ), "contrast swarm contains no batches"
            # Clip the batches to the smaller of the two, so they can be zipped
            batches = min(len(random_compare_batches), len(random_contrast_batches))
            random_contrast_batches = random_compare_batches[:batches]
            random_contrast_batches = random_contrast_batches[:batches]
            random_batches = zip(random_compare_batches, random_contrast_batches)
            for a_batch, b_batch in random_batches:
                a_observs, a_next = a_batch
                b_observs, b_next = b_batch
                step_losses = self.train_step(a_observs, a_next, b_observs, b_next)
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
            print(
                f" - Epoch: {epoch_loss}, AA {loss_a_agreement}, ABD {loss_ab_disagreement}, BB {loss_b_agreement}, BAD {loss_ba_disagreement}"
            )

    def train_step(self, aa, ab, ba, bb):
        with tf.GradientTape() as tape:
            aa = self.model(prepare_batch(aa))
            ab = self.model(prepare_batch(ab))
            ba = self.model(prepare_batch(ba))
            bb = self.model(prepare_batch(bb))
            losses = calculate_contrastive_loss(aa, ab, ba, bb, self.writer)
            a, ab, b, ba = losses
            total_loss = a + ab + b + ba
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights)
            )
        return losses


def peek(iterable) -> Optional[Tuple[Any, Any]]:
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def prepare_batch(batch_observations):
    return numpy.transpose(batch_observations, [0, 2, 1])[:, :, 0]
