import os
import random
from typing import Dict, List, Tuple

import gym
import tensorflow as tf
import tqdm
from wasabi import msg
import mathy as mt
from mathy.state import MathyObservation
from mathy.agents.contrastive import ContrastiveModelTrainer
from mathy.envs import PolySimplify
from mathy.envs.gym import MathyGymEnv

VOCAB = ["", " ", "\t", "\n"] + list(
    ".+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
PAD_TOKEN = VOCAB.index("")
CHAR_TO_INT = {char: index for index, char in enumerate(VOCAB)}
VOCAB_LEN = len(VOCAB)
SEQ_LEN = 256
BATCH_SIZE = 256


def decode_text(tokens: tf.Tensor) -> str:
    """Decode a list of integer tensors to produce a string"""
    output: List[str] = []
    for token in tokens.tolist():
        token_index = int(token)
        assert token_index < VOCAB_LEN, "invalid token"
        output.append(VOCAB[token_index])
    return "".join(output)


def encode_text(text: str, pad_length: int = SEQ_LEN) -> tf.Tensor:
    """Encode text into a list of indices in the vocabulary"""
    values = [CHAR_TO_INT[c] for c in text]
    while len(values) < pad_length:
        values.append(PAD_TOKEN)
    return tf.convert_to_tensor(values, dtype=tf.uint8)


def swap_vars(expression: mt.MathExpression) -> str:
    var_nodes: List[mt.VariableExpression] = expression.find_type(mt.VariableExpression)
    curr_vars: List[str] = list(set([n.identifier for n in var_nodes]))
    new_vars: List[str] = mt.problems.get_rand_vars(
        len(curr_vars), exclude_vars=curr_vars
    )
    var_map = {}
    for var in new_vars:
        var_map[curr_vars.pop()] = var
    node: mt.VariableExpression
    for node in var_nodes:
        node.identifier = var_map[node.identifier]
    return str(expression)


def get_agreement_disagreement_pair(
    env_type: str = "poly", env_difficulty: str = "easy"
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Get one problem text/max_steps tuple"""
    env_name = f"mathy-{env_type}-{env_difficulty}-v0"
    env: MathyGymEnv = gym.make(env_name)

    # HACKS: force selection of expressions with fewer than 6 unique vars
    expression: mt.MathExpression
    while True:
        state, problem = env.mathy.get_initial_state(
            env.env_problem_args, print_problem=False
        )
        expression = env.mathy.parser.parse(problem.text)
        curr_vars: List[str] = list(
            set([n.identifier for n in expression.find_type(mt.VariableExpression)])
        )
        if len(curr_vars) < 6:
            break
    compare: str = problem.text
    compare_aug: str = swap_vars(expression)
    # unlink a node in the tree, and use the remainig subtree as a contrast example
    # NOTE: I'm not sure if this is perfect. The idea is that any subtree changes the
    #       value of an expression, so it should be a fine contrast example... ?
    node = random.choice(
        expression.find_type((mt.AddExpression, mt.MultiplyExpression))
    )
    mt.util.unlink(node.clone())
    contrast: str = str(node)
    contrast_aug = swap_vars(node)

    return (
        encode_text(compare),
        encode_text(compare_aug),
        encode_text(contrast),
        encode_text(contrast_aug),
    )


def make_batch(batch_size=BATCH_SIZE):
    env_types = [
        "poly",
        "binomial",
        "complex",
        "poly-grouping",
        "poly-blockers",
        "poly-combine",
    ]
    aa = []
    ab = []
    ba = []
    bb = []
    for i in range(batch_size):
        env_type = random.choice(env_types)
        caa, cab, cba, cbb = get_agreement_disagreement_pair(env_type)
        aa.append(caa)
        ab.append(cab)
        ba.append(cba)
        bb.append(cbb)

    aa = tf.convert_to_tensor(aa, dtype=tf.float32)
    ab = tf.convert_to_tensor(ab, dtype=tf.float32)
    ba = tf.convert_to_tensor(ba, dtype=tf.float32)
    bb = tf.convert_to_tensor(bb, dtype=tf.float32)
    return aa, ab, ba, bb


if __name__ == "__main__":
    model_file = "training/contrastive/model"
    env: MathyGymEnv = gym.make("mathy-poly-easy-v0")
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
        print(f"Loading checkpoint: {model_file}")
        trainer.model = tf.keras.models.load_model(model_file)
    try:
        trainer.train(batch_fn=make_batch, batches=12, epochs=200)
    except KeyboardInterrupt:
        print("Stopping")
