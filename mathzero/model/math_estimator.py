from .features import FEATURE_NODE_COUNT, FEATURE_TOKEN_VALUES, FEATURE_TOKEN_TYPES


def length(sequence):
    import tensorflow as tf

    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def math_estimator(features, labels, mode, params):
    """TensorFlow custom estimator for RL policy/value network"""
    import tensorflow as tf
    from tensorflow.python.ops import init_ops

    action_size = params["action_size"]

    # Input features
    net = tf.feature_column.input_layer(features, params["feature_columns"])

    # max_length = 512
    # num_hidden = 64

    # token_types_shape = tf.shape(features[FEATURE_TOKEN_TYPES])
    # batch_size = token_types_shape[0]
    # time_steps = tf.reduce_max(token_types_shape[1])
    # sequence_input = tf.convert_to_tensor(features[FEATURE_TOKEN_TYPES], dtype=tf.int32)
    # # sequence_input = tf.reshape(token_types_tensor, shape=[batch_size, time_steps, 1])
    # # sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
    # output, state = tf.nn.dynamic_rnn(
    #     tf.contrib.rnn.GRUCell(num_hidden),
    #     sequence_input,
    #     dtype=tf.int32,
    #     sequence_length=length(sequence_input),
    # )

    training = mode == tf.estimator.ModeKeys.TRAIN
    for units in params["hidden_units"]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    dropout = params.get("dropout", 0.1)
    logits = tf.layers.dense(
        net, action_size, bias_initializer=init_ops.glorot_normal_initializer()
    )

    action_policy = tf.nn.softmax(logits, name="out_policy")
    focus_value = tf.nn.sigmoid(tf.layers.dense(logits, 1), "out_focus")
    value = tf.nn.tanh(tf.layers.dense(logits, 1), "out_value")

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "out_value": value,
            "out_policy": action_policy,
            "out_focus": focus_value,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Split target policies from target values
    target_action_policy = labels[:, 0:-2]
    target_focus = labels[:, -2:-1]
    target_vs = labels[:, -1]

    loss_pi = tf.losses.softmax_cross_entropy(
        target_action_policy, logits, loss_collection="mt"
    )
    metric_loss_pi = tf.summary.scalar("loss_pi", loss_pi)
    loss_v = tf.losses.mean_squared_error(
        target_vs, tf.reshape(value, shape=[-1]), loss_collection="mt"
    )
    loss_focus = tf.losses.mean_squared_error(
        target_focus, focus_value, loss_collection="mt"
    )
    metric_loss_focus = tf.summary.scalar("loss_focus", loss_focus)
    metric_loss_v = tf.summary.scalar("loss_v", loss_v)
    total_loss = loss_pi + loss_v + loss_focus
    # total_loss = loss_pi + loss_v

    # Compute evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss)

    # There's only one option left
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create training op.
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
