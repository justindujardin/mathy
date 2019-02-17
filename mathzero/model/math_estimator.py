def math_estimator(features, labels, mode, params):
    """TensorFlow custom estimator for RL policy/value network"""
    import tensorflow as tf
    from tensorflow.python.ops import init_ops

    action_size = params["action_size"]

    # Input features
    net = tf.feature_column.input_layer(features, params["feature_columns"])
    training = mode == tf.estimator.ModeKeys.TRAIN
    for units in params["hidden_units"]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    dropout = params.get("dropout", 0.1)
    logits = tf.layers.dense(
        net, action_size, bias_initializer=init_ops.glorot_normal_initializer()
    )
    # TODO: add a second policy to have action/position prediction. This should reduce the size considerably
    # because instead of having to predict (action_count * max_input_len) (~768 currently) actions
    # it will have to predict (action_count) actions and then another prediction for (action_position)
    policy = tf.nn.softmax(logits, name="out_policy")
    value = tf.nn.tanh(tf.layers.dense(logits, 1), "out_value")

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"out_value": value, "out_policy": policy, "logits": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Split target policies from target values
    target_pis = labels[:, 0:-1]
    target_vs = labels[:, -1]

    loss_pi = tf.losses.softmax_cross_entropy(target_pis, logits)
    loss_v = tf.losses.mean_squared_error(target_vs, tf.reshape(value, shape=[-1]))
    total_loss = loss_pi + loss_v

    # Compute evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss)

    # There's only one option left
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create training op.
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
