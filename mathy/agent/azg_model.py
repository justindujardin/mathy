from tensorflow.python.ops import init_ops
import tensorflow as tf
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import (
    Input,
    Dense,
    DenseFeatures,
    Dropout,
    TimeDistributed,
)
from tensorflow.python.training import adam
from tensorflow_estimator.contrib.estimator.python import estimator

from ..agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    TENSOR_KEY_VALUE,
)
from ..core.expressions import MathTypeKeysMax
from .layers.bahdanau_attention import BahdanauAttention
from .layers.bi_lstm import BiLSTM
from .layers.densenet_stack import DenseNetStack
from .layers.keras_self_attention import SeqSelfAttention
from .layers.math_policy_dropout import MathPolicyDropout
from .layers.resnet_block import ResNetBlock
from .layers.resnet_stack import ResNetStack


class MathSharedModel:
    def __init__(
        self, action_size: int, learning_rate=3e-4, dropout=0.2, shared_dense_units=32
    ):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.shared_dense_units = shared_dense_units
        self.build_input_tensors()
        self.shared_network = ResNetStack(
            num_layers=2, units=self.shared_dense_units, share_weights=True
        )
        # Push each sequence through the policy layer to predict
        # a policy for each input node. This is a many-to-many prediction
        # where we want to know what the probability of each action is for
        # each node in the expression tree. This is key to allow the model
        # to select which node to apply which action to.
        self.policy_head = TimeDistributed(
            MathPolicyDropout(
                self.action_size,
                dropout=self.dropout,
                feature_layers=[self.shared_network],
            ),
            name="policy_head",
        )
        # Value head
        with tf.compat.v1.variable_scope("value_head"):
            self.value_attn = BahdanauAttention(shared_dense_units)
            self.value_logits = Dense(1, activation="tanh", name="tanh")

        with tf.compat.v1.variable_scope("auxiliary_heads"):
            self.aux_attn = BahdanauAttention(shared_dense_units)
            # Node change prediction
            self.node_ctrl_logits = Dense(1, name="node_ctrl_head")
            # Grouping error prediction
            self.grouping_ctrl_logits = Dense(1, name="grouping_ctrl_head")
            # Group prediction head is an integer value predicting the number
            #  of like-term groups in the observation.
            self.group_prediction_logits = Dense(1, name="group_prediction_head")
            # Reward prediction head with 3 class labels (positive, negative, neutral)
            self.reward_prediction_logits = Dense(3, name="reward_prediction_head")

        # Optimizer (for all tasks)
        self.optimizer = adam.AdamOptimizer(self.learning_rate)

        self.policy_inputs = [
            self.input_bwd_vectors,
            self.input_fwd_vectors,
            self.input_last_rule,
            self.input_move_count,
            self.input_moves_remaining,
            self.input_problem_type,
        ]
        self.policy_model = tf.keras.Model(
            inputs=self.policy_inputs, outputs=self.policy_head
        )

    def build_input_tensors(self):
        # Context inputs
        self.input_move_count = Input(
            name=FEATURE_MOVE_COUNTER, dtype=tf.int64, batch_shape=[None, 1]
        )
        self.input_moves_remaining = Input(
            name=FEATURE_MOVES_REMAINING, dtype=tf.int64, batch_shape=[None, 1]
        )
        self.input_last_rule = Input(
            name=FEATURE_LAST_RULE, dtype=tf.int64, batch_shape=[None, 1]
        )
        self.input_node_count = Input(
            name=FEATURE_NODE_COUNT, dtype=tf.int64, batch_shape=[None, 1]
        )
        self.input_problem_type = Input(
            name=FEATURE_PROBLEM_TYPE, dtype=tf.int64, batch_shape=[None, 1]
        )
        # Sequence inputs
        self.input_bwd_vectors = Input(
            name=FEATURE_BWD_VECTORS, dtype=tf.int64, batch_shape=[None, None, 1]
        )
        self.input_fwd_vectors = Input(
            name=FEATURE_FWD_VECTORS, dtype=tf.int64, batch_shape=[None, None, 1]
        )
        self.input_bwd_last_vectors = Input(
            name=FEATURE_LAST_BWD_VECTORS, dtype=tf.int64, batch_shape=[None, None, 1]
        )
        self.input_fwd_last_vectors = Input(
            name=FEATURE_LAST_FWD_VECTORS, dtype=tf.int64, batch_shape=[None, None, 1]
        )

    def calculate_loss(self):
        import tensorflow as tf

        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(
            self.target_vs, tf.reshape(self.v, shape=[-1])
        )
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(
                self.total_loss
            )


# class MathModel:
#     def __init__(self, game, args):

#         # game params
#         self.board_x, self.board_y = game.get_agent_state_size()
#         self.action_size = game.get_agent_actions_count()
#         self.args = args

#         # Renaming functions
#         Relu = tf.nn.relu
#         Tanh = tf.nn.tanh
#         BatchNormalization = tf.layers.batch_normalization
#         Dropout = tf.layers.dropout
#         Dense = tf.layers.dense

#         # Neural Net
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.input_boards = tf.placeholder(
#                 tf.float32, shape=[None, self.board_x, self.board_y], name="input_state"
#             )  # s: batch_size x board_x x board_y
#             self.dropout = tf.placeholder(tf.float32, name="input_dropout")
#             self.isTraining = tf.placeholder(tf.bool, name="input_training")

#             self.pi = Dense(
#                 self.input_boards,
#                 self.action_size,
#                 bias_initializer=init_ops.glorot_normal_initializer(),
#             )  # batch_size x self.action_size
#             self.prob = tf.nn.softmax(self.pi, name="out_policy")
#             self.v = Tanh(Dense(s_fc2, 1), name="out_value")  # batch_size x 1

#             self.calculate_loss()

#     def conv2d(self, x, out_channels, padding):
#         import tensorflow as tf

#         return tf.layers.conv2d(
#             x, out_channels, kernel_size=[3, 3], padding=padding, use_bias=False
#         )
