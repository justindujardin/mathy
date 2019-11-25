# SCRATCH CELL

import collections
import numpy as np
from mathy.agents.embedding import MathyEmbedding

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

NestedInput = collections.namedtuple("NestedInput", ["feature1", "feature2"])


class FeatureInput(tf.keras.layers.Layer):
    def __init__(
        self,
        nodes: tf.keras.layers.Input,
        mask: tf.keras.layers.Input,
        values: tf.keras.layers.Input,
        type: tf.keras.layers.Input,
        time: tf.keras.layers.Input,
        rnn_state: tf.keras.layers.Input,
        rnn_history: tf.keras.layers.Input,
        **kwargs
    ):
        self.nodes = nodes
        self.mask = mask
        self.values = values
        self.type = type
        self.time = time
        self.rnn_state = rnn_state
        self.rnn_history = rnn_history
        super(FeatureInput, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs


lstm_size = 128
units = 64

unit_1 = 10
unit_2 = 20
unit_3 = 30

input_1 = 32
input_2 = 64
input_3 = 32
batch_size = 64
num_batch = 100
timestep = 50
num_actions = 6

inp_nodes = tf.keras.Input((None, 1), name="nodes_input")
inp_mask = tf.keras.Input((None, num_actions), name="mask_input")
inp_values = tf.keras.Input((None, 1), name="values_input")
inp_type = tf.keras.Input((1,), name="type_input")
inp_time = tf.keras.Input((1,), name="time_input")
inp_rnn_state = tf.keras.Input((2, None, 1, lstm_size), name="rnn_state_input")
inp_rnn_history = tf.keras.Input((2, None, 1, lstm_size), name="rnn_history_input")
inputs = FeatureInput(
    nodes=inp_nodes,
    mask=inp_mask,
    values=inp_values,
    type=inp_type,
    time=inp_time,
    rnn_state=inp_rnn_state,
    rnn_history=inp_rnn_history,
)
model = tf.keras.models.Sequential(
    [
        inp_nodes,
        inp_mask,
        inp_values,
        inp_type,
        inp_time,
        inp_rnn_state,
        inp_rnn_history,
        inputs,
    ]
)
model.run_eagerly = True
model.add(MathyEmbedding(units=units, lstm_units=lstm_size))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], run_eagerly=True)
print(model.summary())
