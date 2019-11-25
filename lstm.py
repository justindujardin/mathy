import datetime
import os
import random
import srsly
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pydantic import BaseModel

# Set random seed for max determinism
random.seed(1337)
np.random.seed(42)
tf.random.set_seed(7)
# Inputs are a Sequence of Sequences of integers
InputsType = List[List[int]]
# Labels are a Sequence of Sequences of floating point 0.0 or 1.0 array values
LabelsType = List[List[List[float]]]


class StateEnum(Enum):
    # Use a trainable state
    train = "train"
    # Zero the state for each batch
    zero = "zero"
    # Persist the state across batches
    persist = "persist"


class WarmupEnum(Enum):
    # Warm up with 0 filled state
    zero = "zero"
    # Warm up by showing the first element in the sequence
    first = "first"


class Args(BaseModel):
    predictions: int = 4
    batches: int = 256
    timesteps: int = 128
    gamma: int = 1
    delta: int = 1
    units: int = 4
    lstm: int = 32
    epochs: int = 20
    warm_up: int = 3
    warm: WarmupEnum = "first"
    state: StateEnum = "train"
    max_int: int = 100
    seq_lstm: bool = False

    class Config:
        extra = "forbid"


def get_log_dir(args: Args) -> str:
    """Generate a dir that includes run hyper parameters in its name"""
    arg_keys: List[str] = []
    for key, value in srsly.json_loads(args.json()).items():
        arg_keys.append(f"{key}{value}")
    arg_keys.sort()
    run_name = "_".join(arg_keys + [datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])
    logdir = os.path.join("logs", run_name)
    return logdir


def build_dataset(args: Args) -> Tuple[InputsType, LabelsType]:
    seen_samples: List[List[int]] = []

    out_examples = np.random.randint(
        args.max_int, size=[args.timesteps, args.predictions]
    ).tolist()
    out_labels = []
    for int_array in out_examples:
        label = []
        for i, v in enumerate(int_array):
            check_sample = None
            if len(seen_samples) >= args.delta:
                check_sample = seen_samples[-args.delta]
            elif len(seen_samples) > 0:
                check_sample = seen_samples[0]
            if check_sample is not None and v > check_sample[i]:
                label.append([1.0])
            else:
                label.append([0.0])
        seen_samples.append(int_array)
        out_labels.append(label)
    return np.array(out_examples), np.array(out_labels)


def build_burn_in_inputs(args: Args) -> InputsType:
    """Build a burn-in sequence"""
    out_examples = []
    init = [0] * args.predictions
    for i in range(args.warm_up):
        out_examples.insert(0, init)
    return np.array(out_examples)


class BatchedDataset(tf.keras.utils.Sequence):
    """Sequence generator that spits out batches of sequences with labels for
    training. Items should not be shuffled because the labels are only valid
    in the order the examples are returned in."""

    def __init__(self, args: Args):
        self.args = args

    def __len__(self):
        return self.args.batches

    def __getitem__(self, idx) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs: InputsType
        labels: LabelsType
        inputs, labels = build_dataset(args)
        # print(inputs)
        # print(labels)
        return tf.convert_to_tensor(inputs), tf.convert_to_tensor(labels)


class ModelBurnInState(tf.keras.callbacks.Callback):
    def __init__(self, args: Args, model: tf.keras.Model, **kwargs):
        super(ModelBurnInState, self).__init__(**kwargs)
        self.args = args
        self.model = model

    def do_burn_in(self):
        if self.args.warm_up > 0:
            inputs = build_burn_in_inputs(self.args)
            self.model.call(inputs)

    def on_train_batch_begin(self, batch, logs=None):
        self.do_burn_in()

    def on_train_batch_end(self, batch, logs=None):
        self.model.embedding.reset_rnn_state()

    def on_test_batch_begin(self, batch, logs=None):
        self.do_burn_in()

    def on_test_batch_end(self, batch, logs=None):
        self.model.embedding.reset_rnn_state()


def train_and_plot(model: tf.keras.Model, args: Args):
    # Build a dataset with random sequences of integers. The challenge is to
    # learn to predict correctly whether the current timestep example
    # elementwise values are greater than or less than the values of the example
    # (delta) steps back in the sequence.
    #
    # Examples = [timestep, None]
    # Labels = [timestep, None, 1]
    # Shape = [timestep, None]
    generator = BatchedDataset(args)
    log_dir = get_log_dir(args)
    print(f"Logdir: {log_dir}")
    model.fit_generator(
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1, write_graph=True,
            ),
            ModelBurnInState(args, model),
        ],
        epochs=args.epochs,
        shuffle=False,
        generator=generator,
        verbose=1,
        workers=4,
    )


class SequenceEmbedding(tf.keras.layers.Layer):
    """Convert a sequence of sequences of integers into a sequence of of
    sequences of fixed-dimension embeddings"""

    def __init__(self, args: Args, **kwargs):
        super(SequenceEmbedding, self).__init__(**kwargs)
        self.lstm_size = args.lstm
        self.args = args
        self.vocab_size = args.max_int + 1
        trainable: bool = args.state == "train"
        self.state_c = tf.Variable(
            tf.zeros([1, self.lstm_size]), trainable=trainable, name="memory/cell",
        )
        self.state_h = tf.Variable(
            tf.zeros([1, self.lstm_size]), trainable=trainable, name="memory/hidden",
        )
        self.embeddings = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=args.units, mask_zero=False,
        )
        self.lstm_batch = tf.keras.layers.LSTM(
            self.lstm_size,
            return_sequences=True,
            time_major=False,
            return_state=True,
            # Props required for GPU:
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
        )
        self.lstm_time = tf.keras.layers.LSTM(
            self.lstm_size,
            return_sequences=True,
            time_major=True,
            return_state=True,
            # Props required for GPU:
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
        )
        self.fully_connected = tf.keras.layers.Dense(args.units, activation="relu")

    def reset_rnn_state(self):
        self.state_h.assign(tf.zeros([1, self.lstm_size]))
        self.state_c.assign(tf.zeros([1, self.lstm_size]))

    def call(self, features: InputsType) -> tf.Tensor:
        inputs = tf.convert_to_tensor(features)
        batch_size = tf.shape(inputs)[0]
        sequence_length = inputs.shape[1]
        outputs = self.embeddings(inputs)
        state_h = self.state_h
        state_c = self.state_c

        # Tile the LSTM state to match the timestep batch sequence length
        if self.args.seq_lstm is True:
            state_h = tf.tile(state_h, [batch_size, 1])
            state_c = tf.tile(state_c, [batch_size, 1])
            outputs, nodes_state_h, nodes_state_c = self.lstm_batch(
                outputs, initial_state=[state_h, state_c]
            )
        state_h = tf.tile(state_h[-1:], [sequence_length, 1])
        state_c = tf.tile(state_c[-1:], [sequence_length, 1])
        outputs, state_h, state_c = self.lstm_time(
            outputs, initial_state=[state_h, state_c]
        )
        if self.args.state != "train":
            self.state_h.assign(state_h[-1:])
            self.state_c.assign(state_c[-1:])
        # Return the embeddings
        return outputs


class RecallAndCompare(tf.keras.Model):
    def __init__(self, args: Args, **kwargs):
        super(RecallAndCompare, self).__init__(name="RecallAndCompare", **kwargs)
        self.units = args.units
        self.predictions = args.predictions
        self.embedding = SequenceEmbedding(args=args, name="embedding")
        self.timestep_dense = tf.keras.layers.Dense(
            self.predictions,
            kernel_initializer="he_normal",
            name="time_dense",
            activation=None,
        )
        self.predictor = tf.keras.layers.TimeDistributed(
            self.timestep_dense, name="predictor",
        )

    def call(self, features: InputsType) -> tf.Tensor:
        embedded = self.embedding(features)
        predicted = self.predictor(embedded)
        return predicted


args = Args(
    units=32,
    lstm=128,
    gamma=1,
    delta=1,
    epochs=10,
    warm_up=3,
    state="persist",
    warm="zero",
    predictions=4,
    batches=16,
    timesteps=64,
)

# Run each configuration multiple times to get average results
print(f"Running: {args.json(indent=2)}")
model = RecallAndCompare(args=args)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True
)
train_and_plot(model, args)
