import tensorflow as tf
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    DenseFeatures,
    TimeDistributed,
    Dropout,
)
from tensorflow.python.training import adam
from tensorflow_estimator.contrib.estimator.python import estimator

from ..features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_VALUE,
)
from ..layers.bahdanau_attention import BahdanauAttention
from ..layers.bi_lstm import BiLSTM
from ..layers.math_policy_dropout import MathPolicyDropout
from ..layers.densenet_stack import DenseNetStack
from ..layers.keras_self_attention import SeqSelfAttention


def math_estimator(features, labels, mode, params):
    sequence_columns = params["sequence_columns"]
    feature_columns = params["feature_columns"]
    action_size = params["action_size"]
    learning_rate = params.get("learning_rate", 0.01)
    dropout_rate = params.get("dropout", 0.2)

    # DenseNet stack configuration
    densenet_layers = params.get("densenet_layers", 6)
    densenet_units = params.get("densenet_units", 128)
    densenet_scaling = params.get("densenet_scaling", 0.75)

    # Self-attention stack configuration
    self_attention_layers = params.get("self_attention_layers", 0)
    self_attention_units = params.get("self_attention_units", 64)

    training = mode == tf.estimator.ModeKeys.TRAIN

    #
    # Sequential/Context feature layers.
    #
    # NOTE: the idea here is that these layers are the least important. I'm not sure
    #  this will work but I'd like to keep the weights from the core of the model while
    #  swapping out the input (and maybe output) layers based on user mods.
    #
    with tf.compat.v1.variable_scope("inputs"):
        sequence_features = {
            FEATURE_BWD_VECTORS: features[FEATURE_BWD_VECTORS],
            FEATURE_FWD_VECTORS: features[FEATURE_FWD_VECTORS],
            FEATURE_LAST_BWD_VECTORS: features[FEATURE_LAST_BWD_VECTORS],
            FEATURE_LAST_FWD_VECTORS: features[FEATURE_LAST_FWD_VECTORS],
        }
        sequence_inputs, sequence_length = SequenceFeatures(
            sequence_columns, name="sequence"
        )(sequence_features)
        features_layer = DenseFeatures(feature_columns, name="context")
        context_inputs = features_layer(features)

    # The core of the model is made with love
    with tf.compat.v1.variable_scope("love"):

        # Bi-directional LSTM over context vectors (with non-sequential features
        # to seed RNN initial_state)
        #
        # Adding these features to the initial state allows us to condition the LSTM
        # on the non-sequence features. Without the context features the policy output
        # will probably struggle to improve beyond a certain point. The reasoning is
        # that the observations are not processed as episode sequences via an RNN of some sort
        # but are input as distinct observations of the environment state. The sequential features
        # are accessed by encoding the previous observation into the input features (via the
        # FEATURE_LAST_FWD_VECTORS and FEATURE_LAST_BWD_VECTORS features). Because of this input
        # format the context features that describe lifetime and urgency (i.e. current_move,
        # moves_remaining) cannot be connected to the sequential policy output predictions.
        #
        hidden_states, sequence_inputs = BiLSTM()(sequence_inputs, context_inputs)
        sequence_inputs = DenseNetStack(
            units=densenet_units,
            num_layers=densenet_layers,
            layer_scaling_factor=densenet_scaling,
        )(sequence_inputs)
        # Apply self-attention to the sequences
        for i in range(self_attention_layers):
            sequence_inputs = SeqSelfAttention(self_attention_units)(sequence_inputs)

        # NOTE: try applying second (smaller) densenet to TimeDistributed

        # Push each sequence through a residual tower and activate it to predict
        # a policy for each input. This is a many-to-many prediction where we want
        # to know what the probability of each action is for each node in the expression
        # tree. This is key to allow the model to select which node to apply which action
        # to.
        predict_policy = TimeDistributed(
            MathPolicyDropout(action_size, dropout=dropout_rate)
        )
        policy_logits = predict_policy(sequence_inputs)
        attention_context, _ = BahdanauAttention(64)(sequence_inputs, hidden_states)
        value_logits = Dense(1, activation="tanh", name="value_logits")(
            attention_context
        )

        # Flatten policy logits
        policy_shape = tf.shape(policy_logits)
        policy_logits = tf.reshape(
            policy_logits, [policy_shape[0], -1, 1], name="policy_reshape"
        )

    logits = {"policy": policy_logits, "value": value_logits}

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    # output histograms for all trainable variables.
    summary_interval = 1000
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
                "value/target_mean", tf.reduce_mean(labels[TRAIN_LABELS_TARGET_VALUE])
            )
            tf.compat.v1.summary.scalar(
                "value/target_variance",
                tf.math.reduce_variance(labels[TRAIN_LABELS_TARGET_VALUE]),
            )
            tf.compat.v1.summary.histogram(
                "policy/target", labels[TRAIN_LABELS_TARGET_PI]
            )
    # Multi-task prediction heads
    with tf.compat.v1.variable_scope("outputs"):
        policy_head = estimator.head.regression_head(name="policy", label_dimension=1)
        value_head = estimator.head.regression_head(name="value", label_dimension=1)
        multi_head = estimator.multi_head.multi_head([policy_head, value_head])
    return multi_head.create_estimator_spec(features, mode, logits, labels, optimizer)
