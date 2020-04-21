"""Implements a contrastive pretraining CLI app for producing math expression
vector representations from character inputs"""
import itertools
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy
import tensorflow as tf
import tqdm
import typer
from fragile.core import HistoryTree, Swarm
from fragile.core.utils import random_state
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from wasabi import msg

from .core.expressions import VariableExpression, MathExpression
from .core.parser import ExpressionParser
from .problems import get_rand_vars
from .envs import PolySimplify
from .envs.gym import MathyGymEnv
from .state import MathyObservation, MathyEnvState

app = typer.Typer()

VOCAB = ["", " ", "\t", "\n"] + list(
    ".+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
PAD_TOKEN = VOCAB.index("")
CHAR_TO_INT = {char: index for index, char in enumerate(VOCAB)}
VOCAB_LEN = len(VOCAB)
SEQ_LEN = 128
BATCH_SIZE = 512
DEFAULT_MODEL = "training/contrastive"


class ContrastiveModelTrainer:
    def __init__(
        self,
        vocab_len: int,
        input_shape: tuple,
        writer: tf.summary.SummaryWriter = None,
        model_file: str = None,
        batch_size: int = 12,
        out_dim: int = 32,
    ):
        self.model_file = model_file
        self.writer = writer
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.model = math_encoder(
            input_shape=input_shape, vocab_len=vocab_len, out_dim=out_dim
        )
        # The input to contrastive prediction task is four double-length hidden vectors
        # from the compare/contrast pairs:
        #
        # A      = self.model(encode_text("4x + 2"))
        # A(aug) = self.model(encode_text("2 + 4x"))
        # B      = self.model(encode_text("7p * p + 2q^4"))
        # B(aug) = self.model(encode_text("7x * x + 2j^4"))
        self.predict = likeness_predictor(input_shape=(batch_size * 4, out_dim * 2))

    def train(
        self, batch_fn: Any, batches: int = 100, epochs: int = 10, verbose: int = 0,
    ):
        for i in range(epochs):
            epoch_loss: float = 0.0
            loss_a_agreement: float = 0.0
            loss_ab_disagreement: float = 0.0
            loss_b_agreement: float = 0.0
            loss_ba_disagreement: float = 0.0
            loss_predict: float = 0.0
            print(f"Iteration: {i}")
            for j in tqdm.tqdm(range(batches)):
                aa_batch, ab_batch, ba_batch, bb_batch = batch_fn()
                step_losses = self.train_step(aa_batch, ab_batch, ba_batch, bb_batch)
                a_agree, ab_disagree, b_agree, ba_disagree, predict = step_losses
                loss_a_agreement += a_agree
                loss_ab_disagreement += ab_disagree
                loss_b_agreement += b_agree
                loss_ba_disagreement += ba_disagree
                loss_predict += predict
                epoch_loss += a_agree + b_agree + ab_disagree + ba_disagree + predict
            if self.writer is not None:
                with self.writer.as_default():
                    step = self.model.optimizer.iterations
                    tf.summary.scalar("loss/predict", loss_predict, step=step)
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
                f"total: {epoch_loss} predict: {loss_predict} aa: {loss_a_agreement} bb: {loss_b_agreement} dab: {loss_ab_disagreement} dba: {loss_ba_disagreement}"
            )

    def train_step(self, aa_batch, ab_batch, ba_batch, bb_batch):
        with tf.GradientTape() as tape:
            aa = self.model(aa_batch)
            ab = self.model(ab_batch)
            ba = self.model(ba_batch)
            bb = self.model(bb_batch)
            # Calculate the cosine similarity loss between our vector pairs
            logits, labels, a, ab, b, ba = calculate_contrastive_loss(
                aa, ab, ba, bb, self.writer
            )
            # Predict the likeness of the a/b b/a pairs returned from the contastive loss
            predict_loss = self.bce(self.predict(logits), labels)
            # Combine the cosine similarity and prediction losses
            total_loss = a + ab + b + ba + predict_loss
            # Update the model
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(
                zip(grads, self.model.trainable_weights)
            )
        return a, ab, b, ba, predict_loss


def get_trainer(folder: str, quiet: bool = False) -> ContrastiveModelTrainer:
    """Create and return a trainer, optionally loading an existing model"""
    model_file = os.path.join(folder, "model")
    log_dir = os.path.join(os.path.dirname(model_file), "tensorboard")
    writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(log_dir)
    trainer = ContrastiveModelTrainer(
        input_shape=(SEQ_LEN,),
        writer=writer,
        model_file=model_file,
        vocab_len=VOCAB_LEN,
        batch_size=BATCH_SIZE,
    )
    if os.path.exists(model_file):
        if not quiet:
            print(f"Loading model: {model_file}")
        trainer.model = tf.keras.models.load_model(model_file)
    return trainer


def calculate_contrastive_loss(
    hidden_aa: tf.Tensor,
    hidden_ab: tf.Tensor,
    hidden_ba: tf.Tensor,
    hidden_bb: tf.Tensor,
    hidden_norm: bool = True,
    writer: tf.summary.SummaryWriter = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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

    # Build logits/labels for prediction task
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


def math_encoder(
    input_shape: Tuple[int, int, int],
    vocab_len: int,
    out_dim: int = 32,
    quiet: bool = False,
):

    layers = [
        tf.keras.layers.Embedding(
            input_shape=input_shape,
            input_dim=vocab_len + 1,
            output_dim=64,
            name="inputs",
            mask_zero=True,
        ),
        Dense(256, activation="relu", name="hidden"),
        Dense(64, activation="relu", name="hidden_2"),
        Flatten(),
        Dense(out_dim, name="output"),
    ]
    model = Sequential(layers)
    model.compile(
        loss="mean_squared_error",
        optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
        metrics=["accuracy"],
    )
    if not quiet:
        model.summary()
    return model


def likeness_predictor(input_shape: Tuple[int, int, int]):
    """Predict if two vectors are different views of the same expression
    or views of different expressions"""
    model = Sequential(
        [
            Dense(512, activation="relu", name="head", batch_input_shape=input_shape),
            Dense(1, name="output"),
        ]
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
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
    values = [CHAR_TO_INT[c] for c in text]
    while len(values) < pad_length:
        values.append(PAD_TOKEN)
    if include_batch:
        values = [values]
    return tf.convert_to_tensor(values, dtype=tf.float32)


#
# Data augmentation
#


def swap_vars(expression: MathExpression) -> str:
    """Given an input expression, substitute unique variables in for the 
    ones in the expression. This augmentation works because by mapping the
    existing variables to others, we maintain the value of the expression if
    it were to be evaluated. We can say that it's a view of the same expression
    because the value hasn't changed.
    
    # Examples

    - "4x + 2x" => "4y + 2y"
    - "12b^3 - 7x + 14.3y" => "12i^3 - 7l + 14.3q"    
    """
    var_nodes: List[VariableExpression] = expression.find_type(VariableExpression)
    curr_vars: List[str] = list(set([n.identifier for n in var_nodes]))
    new_vars: List[str] = get_rand_vars(len(curr_vars), exclude_vars=curr_vars)
    var_map = {}
    for var in new_vars:
        var_map[curr_vars.pop()] = var
    node: VariableExpression
    for node in var_nodes:
        node.identifier = var_map[node.identifier]
    return str(expression)


def augment_problem(expression: MathExpression, env: MathyGymEnv) -> str:
    roll = random.random()
    if roll < 0.3:
        # Swap variables leaving the same expression
        augmented = swap_vars(expression)
    elif roll < 0.6:
        # Apply random actions
        state = MathyEnvState(problem=str(expression))
        for i in range(random.randint(0, 4)):
            try:
                action = env.mathy.random_action(expression)
            except ValueError:
                # Happens if there are no valid actions (accidentally solved?)
                break
            state: MathyEnvState = env.mathy.get_next_state(state, action)[0]
            expression = env.mathy.parser.parse(state.agent.problem)
        augmented = str(expression)
    else:
        # Remove spaces randomly
        def space_indices(s) -> List[int]:
            return [i for i, ltr in enumerate(augmented) if ltr == " "]

        augmented = str(expression)
        spaces = space_indices(augmented)
        for i in range(random.randint(1, len(spaces))):
            spaces = space_indices(augmented)
            if len(spaces) == 0:
                continue
            space_idx = random.choice(spaces)
            augmented = augmented[:space_idx] + augmented[space_idx + 1 :]

    # Finally, roll to see if random spaces should be prepended or appended
    if random.random() > 0.5:
        augmented = (" " * random.randint(1, 5)) + augmented
    if random.random() > 0.5:
        augmented = augmented + (" " * random.randint(1, 5))
    return augmented


#
# Dataset generation
#


def generate_problem(env: MathyGymEnv) -> str:
    # HACKS: force selection of expressions with fewer than 6 unique vars
    compare_exp: MathExpression
    while True:
        state, problem = env.mathy.get_initial_state(
            env.env_problem_args, print_problem=False
        )
        compare_exp = env.mathy.parser.parse(problem.text)
        curr_vars: List[str] = list(
            set([n.identifier for n in compare_exp.find_type(VariableExpression)])
        )
        if len(curr_vars) < 6:
            break
    return problem.text


def get_agreement_disagreement_pair(
    env_types: List[str], env_difficulty: str = "easy"
) -> Tuple[List[int], List[int], List[int], List[int]]:
    parser: ExpressionParser = ExpressionParser()
    env_difficulty = random.choice(["easy", "normal"])
    candidate_types: List[str] = env_types[:]
    random.shuffle(candidate_types)
    compare_env: MathyGymEnv = gym.make(
        f"mathy-{candidate_types[0]}-{env_difficulty}-v0"
    )
    contrast_env: MathyGymEnv = gym.make(
        f"mathy-{candidate_types[-1]}-{env_difficulty}-v0"
    )
    compare: str = generate_problem(compare_env)
    compare_aug: str = augment_problem(
        compare_env.mathy.parser.parse(compare), env=compare_env
    )
    contrast: str = generate_problem(contrast_env)
    contrast_aug: str = augment_problem(
        compare_env.mathy.parser.parse(contrast), env=compare_env
    )
    # if random.random() > 0.99:
    #     print(f"pos: {compare}")
    #     print(f"pos: {compare_aug}")
    #     print(f"neg: {contrast}")
    #     print(f"neg: {contrast_aug}")
    return (
        encode_text(compare),
        encode_text(compare_aug),
        encode_text(contrast),
        encode_text(contrast_aug),
    )


def make_batch(batch_size=BATCH_SIZE):
    env_types = ["poly", "binomial", "complex"]
    aa = []
    ab = []
    ba = []
    bb = []
    for i in range(batch_size):
        caa, cab, cba, cbb = get_agreement_disagreement_pair(env_types)
        aa.append(caa)
        ab.append(cab)
        ba.append(cba)
        bb.append(cbb)

    aa = tf.convert_to_tensor(aa, dtype=tf.float32)
    ab = tf.convert_to_tensor(ab, dtype=tf.float32)
    ba = tf.convert_to_tensor(ba, dtype=tf.float32)
    bb = tf.convert_to_tensor(bb, dtype=tf.float32)
    return aa, ab, ba, bb


@app.command()
def train(
    folder: str = typer.Argument(DEFAULT_MODEL),
    eval_every: int = 4,
    iterations: int = 500,
):
    trainer: ContrastiveModelTrainer = get_trainer(folder)
    trainer.train(batch_fn=make_batch, batches=eval_every, epochs=iterations)


@app.command()
def compare(first: str, second: str, model: str = DEFAULT_MODEL):
    def cosine_similarity(a: numpy.ndarray, b: numpy.ndarray):
        return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

    trainer: ContrastiveModelTrainer = get_trainer(model)
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
