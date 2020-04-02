import pytest
import numpy as np
from mathy.agents.densenet import DenseNetBlock, DenseNetStack
import tensorflow as tf


def test_densenet_construction():
    layer: DenseNetStack = DenseNetStack()
    assert layer is not None
    model = tf.keras.Sequential([layer])
    model(np.zeros(shape=(128, 1)))


def test_densenet_errors():
    with pytest.raises(ValueError):
        DenseNetStack(normalization_style="invalid")


@pytest.mark.parametrize("norm_type", ("batch", "layer"))
def test_densenet_normalization_types(norm_type: str):
    model = tf.keras.Sequential(
        [DenseNetStack(units=24, num_layers=4, normalization_style=norm_type)]
    )
    y = model(np.zeros(shape=(10, 1)))


def test_densenet_no_activation():
    model = tf.keras.Sequential(
        [DenseNetStack(units=24, num_layers=4, activation=None)]
    )
    y = model(np.zeros(shape=(10, 1)))
    assert y.shape[0] == 10


def test_densenet_output_transform():
    layer: DenseNetStack = DenseNetStack(
        units=10,
        num_layers=4,
        activation=None,
        output_transform=tf.keras.layers.Dense(128),
    )
    assert layer is not None
    model = tf.keras.Sequential([layer])
    x = np.zeros(shape=(10, 1))
    y = model(x)
    assert y.shape[1] == 128


def test_densenet_share_weights():
    layer: DenseNetStack = DenseNetStack(share_weights=True)
    layer_two: DenseNetStack = DenseNetStack(share_weights=True)
    assert layer.get_weights() == layer_two.get_weights()
    model = tf.keras.Sequential([layer, layer_two])
    x = np.zeros(shape=(10, 1))
    y = model(x)

