import tensorflow as tf
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import Concatenate, Dense, DenseFeatures, TimeDistributed
from tensorflow.python.training import adam
from tensorflow_estimator.contrib.estimator.python import estimator

from ..agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    TRAIN_LABELS_TARGET_PI,
    TRAIN_LABELS_TARGET_VALUE,
)
from .layers.bahdanau_attention import BahdanauAttention
from .layers.bi_lstm import BiLSTM
from .layers.math_policy import MathPolicy
from .layers.residual_dense import ResidualDense


def math_estimator(features, labels, mode, params):
    sequence_columns = params["sequence_columns"]
    feature_columns = params["feature_columns"]
    action_size = params["action_size"]
    learning_rate = params.get("learning_rate", 0.01)
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

    # The core of the model is our feature extraction/processing pipeline.
    # These are the weights we transfer when there are incompatible input/output
    # architectures.
    with tf.compat.v1.variable_scope("love"):
        # Context feature residual tower
        for i in range(6):
            context_inputs = ResidualDense(4)(context_inputs)

        # Bi-directional LSTM over context vectors
        hidden_states, sequence_inputs = BiLSTM(12)(sequence_inputs)

        # Push each sequence through a residual tower and activate it to predict
        # a policy for each input. This is a many-to-many prediction where we want
        # to know what the probability of each action is for each node in the expression
        # tree. This is key to allow the model to select which node to apply which action
        # to.
        predict_policy = TimeDistributed(MathPolicy(action_size))
        sequence_outputs = predict_policy(sequence_inputs)

        # TODO: I can't get the reshape working to mix the context-features with the sequence
        #  features before doing the softmax prediction above. Without these features I think
        #  the policy output will struggle to improve beyond a certain point. The reasoning is
        #  that the observations are not processed as episode sequences via an RNN of some sort
        #  but are input as distinct observations of the environment state. The sequential features
        #  are accessed by encoding the previous observation into the input features (via the
        #  FEATURE_LAST_FWD_VECTORS and FEATURE_LAST_BWD_VECTORS features). Because of this input
        #  format the context features that describe lifetime and urgency (i.e. current_move,
        #  moves_remaining) cannot be connected to the sequential policy output predictions.
        # TODO: Someone help! ☝️ Thanks! 🙇‍

        attention_context, _ = BahdanauAttention(4096)(sequence_outputs, hidden_states)
        network = Concatenate(name="mixed_features")(
            [attention_context, context_inputs]
        )
        value_logits = Dense(1, activation="tanh", name="value_logits")(network)

        # Flatten policy logits
        policy_logits = sequence_outputs
        policy_shape = tf.shape(policy_logits)
        policy_logits = tf.reshape(
            policy_logits, [policy_shape[0], -1, 1], name="policy_reshape"
        )

    logits = {"policy": policy_logits, "value": value_logits}

    # Optimizer (for all tasks)
    optimizer = adam.AdamOptimizer(learning_rate)

    # output histograms for all trainable variables.
    summary_interval = 500
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
