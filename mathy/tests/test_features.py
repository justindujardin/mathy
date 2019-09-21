from ..features import (
    MathyBatchObservationFeatures,
    MathyObservationFeatures,
    MathyBatchWindowObservationFeatures,
)
from ..state import MathyEnvState


def test_mathy_features_from_state():
    state = MathyEnvState()

    state.to_input_features(state.move_mask)
    state = MathyEnvState()
