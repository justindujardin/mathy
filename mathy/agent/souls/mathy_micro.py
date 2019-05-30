import tensorflow as tf
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import Dense, DenseFeatures, TimeDistributed
from tensorflow.python.training import adam
from tensorflow_estimator.contrib.estimator.python import estimator

from ..features import (
    FEATURE_BWD_VECTORS,  # FEATURE_MOVE_MASK,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    TENSOR_KEY_VALUE,
)
from ..layers.bahdanau_attention import BahdanauAttention
from ..layers.math_policy_dropout import MathPolicyDropout
from ..layers.resnet_stack import ResNetStack
from ..layers.keras_self_attention import SeqSelfAttention
from ..layers.bi_lstm import BiLSTM
from ..layers.lstm_stack import LSTMStack


def math_estimator(features, labels, mode, params):
    sequence_columns = params["sequence_columns"]
    feature_columns = params["feature_columns"]
    action_size = params["action_size"]
    learning_rate = params["learning_rate"]
    dropout_rate = params["dropout"]
    shared_dense_units = params.get("shared_dense_units", 128)
    sequence_features = {
        # FEATURE_MOVE_MASK: features[FEATURE_MOVE_MASK],
        FEATURE_BWD_VECTORS: features[FEATURE_BWD_VECTORS],
        FEATURE_FWD_VECTORS: features[FEATURE_FWD_VECTORS],
        FEATURE_LAST_BWD_VECTORS: features[FEATURE_LAST_BWD_VECTORS],
        FEATURE_LAST_FWD_VECTORS: features[FEATURE_LAST_FWD_VECTORS],
    }
    # Pop the policy mask off (since we use it directly)
    # pi_mask = features[FEATURE_MOVE_MASK]
    # del features[FEATURE_MOVE_MASK]

    sequence_inputs, sequence_length = SequenceFeatures(
        sequence_columns, name="seq_features"
    )(sequence_features)
    context_inputs = DenseFeatures(feature_columns, name="ctx_features")(features)
    lstm = LSTMStack(units=shared_dense_units, share_weights=True)
    shared_network = ResNetStack(shared_dense_units, share_weights=True)
    hidden_states, lstm_vectors = lstm(sequence_inputs, context_inputs)

    # Push each sequence through the policy layer to predict
    # a policy for each input node. This is a many-to-many prediction
    # where we want to know what the probability of each action is for
    # each node in the expression tree. This is key to allow the model
    # to select which node to apply which action to.
    policy_head = TimeDistributed(
        MathPolicyDropout(action_size, dropout=dropout_rate, feature_layers=[]),
        name="policy_head",
    )
    policy_predictions = policy_head(lstm_vectors)

    # Value head
    with tf.compat.v1.variable_scope("value_head"):
        attention_context, attention_weights = BahdanauAttention(shared_dense_units)(
            shared_network(lstm_vectors), hidden_states
        )
        value_logits = Dense(1, activation="tanh", name="tanh")(attention_context)

    with tf.compat.v1.variable_scope("auxiliary_heads"):

        aux_attention, aux_attention_weights = BahdanauAttention(shared_dense_units)(
            shared_network(lstm_vectors), hidden_states
        )
        # Node change prediction
        node_ctrl_logits = Dense(1, name="node_ctrl_head")(aux_attention)
        # Grouping error prediction
        grouping_ctrl_logits = Dense(1, name="grouping_ctrl_head")(aux_attention)
        # Group prediction head is an integer value predicting the number
        #  of like-term groups in the observation.
        group_prediction_logits = Dense(1, name="group_prediction_head")(aux_attention)
        # Reward prediction head with 3 class labels (positive, negative, neutral)
        reward_prediction_logits = Dense(3, name="reward_prediction_head")(
            aux_attention
        )

    logits = {
        "policy": policy_predictions,
        "value": value_logits,
        "node_ctrl": node_ctrl_logits,
        "grouping_ctrl": grouping_ctrl_logits,
        "reward_prediction": reward_prediction_logits,
        "group_prediction": group_prediction_logits,
    }

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    # output histograms for all trainable variables.
    summary_interval = 100
    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.name, var)

    with tf.compat.v1.variable_scope("stats"):

        # Output values
        tf.compat.v1.summary.scalar("value/mean", tf.reduce_mean(value_logits))
        tf.compat.v1.summary.scalar(
            "value/variance", tf.math.reduce_variance(value_logits)
        )
        # Training targets
        if labels is not None:
            tf.compat.v1.summary.scalar(
                "value/target_mean", tf.reduce_mean(labels[TENSOR_KEY_VALUE])
            )
            tf.compat.v1.summary.scalar(
                "value/target_variance",
                tf.math.reduce_variance(labels[TENSOR_KEY_VALUE]),
            )
    # Multi-task prediction heads
    with tf.compat.v1.variable_scope("heads"):
        policy_head = estimator.head.regression_head(
            name="policy", label_dimension=action_size
        )
        value_head = estimator.head.regression_head(name="value", label_dimension=1)
        aux_node_ctrl_head = estimator.head.regression_head(
            name="node_ctrl", label_dimension=1
        )
        aux_grouping_ctrl_head = estimator.head.regression_head(
            name="grouping_ctrl", label_dimension=1
        )
        aux_group_prediction_head = estimator.head.regression_head(
            name="group_prediction", label_dimension=1
        )
        aux_reward_prediction_head = estimator.head.regression_head(
            name="reward_prediction", label_dimension=3
        )
        heads = [
            policy_head,
            value_head,
            aux_node_ctrl_head,
            aux_grouping_ctrl_head,
            aux_group_prediction_head,
            aux_reward_prediction_head,
        ]
        # The first two (policy/value) heads get full weight, and the aux
        # tasks combine to half the weight of the policy/value heads
        # head_weights = [1.0, 1.0, 0.25, 0.75, 0.5, 0.5]
        multi_head = estimator.multi_head.multi_head(heads)
    return multi_head.create_estimator_spec(features, mode, logits, labels, optimizer)
