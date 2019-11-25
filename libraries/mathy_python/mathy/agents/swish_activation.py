import tensorflow as tf


def swish(x):
    """Swish activation function: https://arxiv.org/pdf/1710.05941.pdf"""
    return x * tf.nn.sigmoid(x)
