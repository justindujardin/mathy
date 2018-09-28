from ..environment_state import EnvironmentState
from ..core.parser import ExpressionParser


def test_math_state():
    state = EnvironmentState(128)
    assert state is not None


def test_math_state_encode_player():
    state = EnvironmentState(128)
    parser = ExpressionParser()
    env_state = state.encode_board("4x+2")
    env_state = state.encode_player(env_state, 1, parser.make_features("2+4x"), 10)
    features, move_count, _ = state.decode_player(env_state, 1)
    assert str(state.parser.parse_features(features)).strip() == "2 + 4x"
    assert move_count == 10


def test_math_state_decode_player():
    state = EnvironmentState(128)
    for player_id in (1, -1):
        env_state = state.encode_board("4x+2")
        _, move_count, player_index = state.decode_player(env_state, player_id)
        assert move_count == 0
        assert player_id == player_index


def test_math_state_get_canonical_board():
    state = EnvironmentState(128)
    parser = ExpressionParser()
    env_state = state.encode_board("4x+2")
    env_state = state.encode_player(env_state, -1, parser.make_features("2+4x"), 1)
    env_state = state.get_canonical_board(env_state, -1)
    # The canonical env_state always represents the env_state from the same perspective, in this case
    # from the perspective of player 1. So player -1's canonical env_state will return the player -1
    # state when you decode player 1 from it.
    _, move_count, player_index = state.decode_player(env_state, 1)
    assert move_count == 1
    assert player_index == -1
