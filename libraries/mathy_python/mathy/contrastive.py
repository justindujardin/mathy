"""Implements a contrastive pretraining CLI app for producing math expression
vector representations from character inputs"""
import os
import random
from typing import Any, List, Tuple

import gym
import numpy
import tensorflow as tf
import tqdm
import typer
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import CustomObjectScope

from .agents.attention import SeqSelfAttention
from .agents.densenet import DenseNetStack
from .core.expressions import (
    AddExpression,
    DivideExpression,
    MathExpression,
    MultiplyExpression,
    SubtractExpression,
)
from .envs.gym import MathyGymEnv
from .problems import use_pretty_numbers
from .state import MathyEnvState

app = typer.Typer()

VOCAB = ["", " ", "\t", "\n"] + list(
    ".+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
PAD_TOKEN = VOCAB.index("")
CHAR_TO_INT = {char: index for index, char in enumerate(VOCAB)}
VOCAB_LEN = len(VOCAB)
SEQ_LEN = 128
BATCH_SIZE = 128
DEFAULT_MODEL = "training/contrastive"
LARGE_NUM = 1e9

# Optimizer
LR_INITIAL = 0.001
LR_DECAY_STEPS = 50
LR_DECAY_RATE = 0.96
LR_DECAY_STAIRCASE = False


# Reuse the augmentation envs
GYM_ENVS: List[MathyGymEnv] = []
for name in ["poly", "complex", "binomial"]:
    GYM_ENVS.append(gym.make(f"mathy-{name}-easy-v0"))


class ContrastiveModelTrainer:
    def __init__(
        self,
        vocab_len: int,
        input_shape: Tuple[int, ...],
        writer: tf.summary.SummaryWriter = None,
        model_file: str = None,
        project_file: str = None,
        batch_size: int = 12,
        out_dim: int = 128,
        embed_dim: int = 1024,
        training: bool = True,
        profile: bool = False,
    ):
        self.model_file = model_file
        self.project_file = project_file
        self.writer = writer
        self.training = training
        self.profile = profile
        self.model = build_math_encoder(
            input_shape=input_shape,
            vocab_len=vocab_len,
            out_dim=out_dim,
            embed_dim=embed_dim,
        )
        self.project = build_projection_network(input_shape=(out_dim,))

    def train(
        self, batch_fn: Any, batches: int = 100, epochs: int = 10,
    ):

        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.trace_on(graph=True)

                @tf.function
                def trace_fn():
                    self.model(encode_text("4x+2x", include_batch=True))

                trace_fn()
                tf.summary.trace_export(
                    name="model",
                    step=0,
                    profiler_outdir=os.path.dirname(str(self.model_file)),
                )
                tf.summary.trace_off()

        pr = None
        if self.profile:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()
            print("PROFILER: recording")

        try:
            self.train_loop(batch_fn=batch_fn, epochs=epochs, batches=batches)
        except KeyboardInterrupt:
            print("Exiting...")

        if self.profile:
            assert pr is not None
            profile_path = os.path.join(
                os.path.dirname(f"{self.model_file}"), f"training.profile"
            )
            pr.disable()
            pr.dump_stats(profile_path)
            print(f"PROFILER: saved {profile_path}")

    def train_loop(self, batch_fn: Any, batches: int = 100, epochs: int = 10):
        for i in range(epochs):
            epoch_loss: List[float] = []
            epoch_acc: List[float] = []
            epoch_entropy: List[float] = []
            print(f"Iteration: {i}")
            for j in tqdm.tqdm(range(batches)):
                # start = time.time()
                batch = batch_fn()
                # print(f"make batch: {time.time() - start}")
                # start = time.time()
                contrast_loss, contrast_acc, contrast_entropy = self.train_step(batch)
                # print(f"train step: {time.time() - start}")
                epoch_acc.append(contrast_acc)
                epoch_loss.append(contrast_loss)
                epoch_entropy.append(contrast_entropy)
            if self.writer is not None:
                with self.writer.as_default():
                    step = self.model.optimizer.iterations
                    tf.summary.scalar(
                        f"contrast/learning_rate",
                        data=self.model.optimizer.lr(step),
                        step=step,
                    )
                    tf.summary.scalar(
                        "contrast/loss", tf.reduce_mean(epoch_loss), step=step
                    )
                    tf.summary.scalar(
                        "contrast/entropy", tf.reduce_mean(epoch_entropy), step=step
                    )
                    tf.summary.scalar(
                        "contrast/accuracy", tf.reduce_mean(epoch_acc), step=step
                    )
                    # representation variables
                    for var in self.model.trainable_variables:
                        tf.summary.histogram(var.name, var, step=step)
                    # projection variables
                    for var in self.project.trainable_variables:
                        tf.summary.histogram(var.name, var, step=step)

                tf.keras.models.save_model(self.model, self.model_file)
                tf.keras.models.save_model(self.project, self.project_file)
            print(
                f"loss: {tf.reduce_mean(epoch_loss)} acc: {tf.reduce_mean(epoch_acc)} entropy: {tf.reduce_mean(epoch_entropy)}"
            )

    def train_step(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            batch_a, batch_b = tf.split(batch, 2, -1)
            hiddens_a = self.project(self.model(batch_a))
            hiddens_b = self.project(self.model(batch_b))
            hiddens = tf.concat([hiddens_a, hiddens_b], axis=-1)
            # Calculate the cosine similarity loss between our vector pairs
            total_loss, logits_con, labels_con = add_contrastive_loss(hiddens)
            # Compute stats for the summary.
            prob_con = tf.nn.softmax(logits_con)
            entropy_con = -tf.reduce_mean(
                tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1)
            )
            contrast_acc = tf.equal(
                tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1)
            )
            contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))
        # Update the representation model
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Update the projection model
        pred_grads = tape.gradient(total_loss, self.project.trainable_weights)
        pred_grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in pred_grads]
        self.project.optimizer.apply_gradients(
            zip(pred_grads, self.project.trainable_weights)
        )
        #
        # Debug gradient norms
        #
        # print(f"repr - {tf.linalg.global_norm(grads).numpy()}")
        # print(f"pred - {tf.linalg.global_norm(pred_grads).numpy()}")

        # Don't forget to free the persistent=True tape
        del tape
        return total_loss, contrast_acc, entropy_con


def get_trainer(
    folder: str, quiet: bool = False, training: bool = True, profile: bool = False
) -> ContrastiveModelTrainer:
    """Create and return a trainer, optionally loading an existing model"""
    model_file = os.path.join(folder, "representation")
    project_file = os.path.join(folder, "project")
    log_dir = os.path.join(os.path.dirname(model_file), "tensorboard")
    writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(log_dir)
    trainer = ContrastiveModelTrainer(
        input_shape=(SEQ_LEN,),
        writer=writer,
        model_file=model_file,
        project_file=project_file,
        vocab_len=VOCAB_LEN,
        batch_size=BATCH_SIZE,
        training=training,
        profile=profile,
    )
    if os.path.exists(model_file):
        if not quiet:
            print(f"Loading representation: {model_file}")
        with CustomObjectScope(
            {"SeqSelfAttention": SeqSelfAttention, "DenseNetStack": DenseNetStack}
        ):
            trainer.model = tf.keras.models.load_model(model_file)
    if os.path.exists(project_file):
        if not quiet:
            print(f"Loading predictor: {project_file}")
        trainer.project = tf.keras.models.load_model(project_file)
    if not quiet:
        trainer.model.summary()
        trainer.project.summary()
    return trainer


def swish(x):
    """Swish activation function: https://arxiv.org/pdf/1710.05941.pdf"""
    return x * tf.nn.sigmoid(x)


def build_exponential_decay_adam(
    lr_initial: float = LR_INITIAL,
    lr_decay_steps: float = LR_DECAY_STEPS,
    lr_decay_rate: float = LR_DECAY_RATE,
    lr_decay_staircase: bool = LR_DECAY_STAIRCASE,
) -> tf.keras.optimizers.Optimizer:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr_initial,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate,
        staircase=lr_decay_staircase,
    )
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)


def build_math_encoder(
    input_shape: Tuple[int, ...],
    vocab_len: int,
    embed_dim: int,
    out_dim: int,
    quiet: bool = False,
):
    layers = [
        tf.keras.layers.Embedding(
            input_shape=input_shape,
            input_dim=vocab_len + 1,
            output_dim=embed_dim,
            name="embeddings",
            mask_zero=True,
        ),
        tf.keras.layers.LSTM(out_dim, name="lstm", return_sequences=False),
        tf.keras.layers.LayerNormalization(name="layer_norm"),
    ]
    model = Sequential(layers, name="representation")
    model.compile(
        loss="mean_squared_error",
        optimizer=build_exponential_decay_adam(),
        metrics=["accuracy"],
    )
    return model


def build_projection_network(input_shape: Tuple[int, ...]) -> tf.keras.Sequential:
    """Projection model for building a richer representation of output vectors for the
    contrastive loss task. The paper claims this greatly improved performance."""
    model = Sequential(
        [
            Dense(
                64, activation="relu", name="projection/head", input_shape=input_shape,
            ),
            tf.keras.layers.LayerNormalization(name="projection/head_ln"),
            Dense(32, name="projection/output"),
        ],
        name="projection",
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=build_exponential_decay_adam(),
        metrics=["accuracy"],
    )
    return model


#
# Text encoding/decoding
#


def decode_text(tokens: tf.Tensor) -> str:
    """Decode a list of integer tensors to produce a string"""
    output: List[str] = []
    for token in tokens.tolist():
        token_index = int(token)
        assert token_index < VOCAB_LEN, "invalid token"
        output.append(VOCAB[token_index])
    return "".join(output)


def encode_text(
    text: str, pad_length: int = SEQ_LEN, include_batch: bool = False
) -> tf.Tensor:
    """Encode text into a list of indices in the vocabulary"""
    values = [CHAR_TO_INT[c] for c in text] + [PAD_TOKEN] * (pad_length - len(text))
    if include_batch:
        values = [values]
    return tf.convert_to_tensor(values, dtype=tf.float32)


#
# Data augmentation
#


def some_spaces(min_inclusive: int = 0, max_exclusive: int = 3) -> str:
    return " " * random.randint(min_inclusive, max_exclusive)


def space_indices(s) -> List[int]:
    return [i for i, ltr in enumerate(s) if ltr == " "]


def augment_problem(
    expression: MathExpression,
    *,
    env: MathyGymEnv,
    min_transforms: int = 1,
    max_transforms: int = 3,
    can_drop: bool = True,
) -> str:
    augmented = str(expression)
    if random.random() > 0.85 and can_drop:
        # Drop part of the tree and return a subtree. This breaks the value evaluation
        # of the expressions (if that's the comparison metric you want to use) but is
        # similar to a crop in an image.
        operators = expression.find_type(
            (AddExpression, MultiplyExpression, DivideExpression, SubtractExpression)
        )
        if len(operators) > 0:
            target: MathExpression = random.choice(operators)
            augmented = str(target)
    else:
        # Apply random actions
        state = MathyEnvState(problem=augmented)
        for _ in range(random.randint(1, max_transforms)):
            try:
                action = env.mathy.random_action(expression)
            except ValueError:
                # Happens if there are no valid actions (accidentally solved?)
                break
            state: MathyEnvState = env.mathy.get_next_state(state, action)[0]
            expression = env.mathy.parser.parse(state.agent.problem)
        augmented = str(expression)

    # Remove spaces randomly
    spaces = space_indices(augmented)
    if len(spaces) > 0:
        steps = random.randint(1, len(spaces))
        for i in range(steps):
            spaces = space_indices(augmented)
            if len(spaces) == 0:
                break
            space_idx = random.choice(spaces)
            # Either remove the space or add a few extra
            mid = some_spaces(2, 4) if random.random() > 0.7 else ""
            augmented = augmented[:space_idx] + mid + augmented[space_idx + 1 :]

    # Finally maybe append/prepend spaces
    augmented = some_spaces() + augmented + some_spaces()
    return augmented


#
# Dataset generation
#


def generate_problem(env: MathyGymEnv) -> str:
    _, problem = env.mathy.get_initial_state(env.env_problem_args, print_problem=False)
    return problem.text


def get_agreement_pair(envs: List[MathyGymEnv]) -> List[int]:
    env: MathyGymEnv = random.choice(envs)
    problem: str = generate_problem(env)
    augment_one: str = augment_problem(env.mathy.parser.parse(problem), env=env)
    # Don't allow dropping parts of the tree on the second example to avoid invalid
    # augmentations where both are local views of some global context, but neither
    # has enough information to connect the two. Unlike an image of a dog, you can't
    # necessarily determine that non-overlapping chunks of the same expression
    # are from the same source (unless the source is visible)
    augment_two: str = augment_problem(
        env.mathy.parser.parse(problem), env=env, can_drop=False
    )
    # if random.random() > 0.99:
    #     print(f"text: {problem}")
    #     print(f"aug1: {augment_one}")
    #     print(f"aug2: {augment_two}")
    return tf.concat([encode_text(augment_one), encode_text(augment_two)], axis=-1)


def make_batch(batch_size=BATCH_SIZE) -> tf.Tensor:
    global GYM_ENVS
    pairs = [get_agreement_pair(GYM_ENVS) for _ in range(batch_size)]
    pairs_tensor = tf.convert_to_tensor(pairs, dtype=tf.float32)
    return pairs_tensor


def add_contrastive_loss(
    hidden: tf.Tensor, hidden_norm: bool = True, temperature=0.1, weights=1.0
):
    """Compute NT-Xent contrastive loss.

    Copyright 2020 The SimCLR Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific simclr governing permissions and
    limitations under the License.
    ==============================================================================    

    Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    weights: a weighting number or vector.
    Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, -1)
    batch_size = tf.shape(hidden1)[0]

    # Gather hidden1/hidden2 across replicas and create local labels.
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], 1), weights=weights
    )
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1), weights=weights
    )
    loss = loss_a + loss_b

    return loss, logits_ab, labels


#
# CLI App
#


@app.command()
def train(
    folder: str = typer.Argument(DEFAULT_MODEL),
    eval_every: int = 4,
    iterations: int = 500,
    profile: bool = False,
):
    use_pretty_numbers(False)
    trainer: ContrastiveModelTrainer = get_trainer(folder, profile=profile)
    trainer.train(batch_fn=make_batch, batches=eval_every, epochs=iterations)


@app.command()
def compare(first: str, second: str, model: str = DEFAULT_MODEL):
    def cosine_similarity(a: numpy.ndarray, b: numpy.ndarray):
        return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

    trainer: ContrastiveModelTrainer = get_trainer(model, quiet=True, training=False)
    first_repr: numpy.ndarray = tf.squeeze(
        trainer.model(encode_text(first, include_batch=True))
    ).numpy()
    second_repr: numpy.ndarray = tf.squeeze(
        trainer.model(encode_text(second, include_batch=True))
    ).numpy()
    similarity = cosine_similarity(first_repr, second_repr)
    typer.echo(f'similarity("{first}", "{second}") = {similarity.tolist()}')


if __name__ == "__main__":
    app()
