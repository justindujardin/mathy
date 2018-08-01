from ..environment_state import EnvironmentState
from ..core.parser import ExpressionParser


def test_math_state():
    state = EnvironmentState(128)
    assert state is not None


def test_math_state_encode_player():
    state = EnvironmentState(128)
    parser = ExpressionParser()
    board = state.encode_board("4x+2")
    board = state.encode_player(board, 1, parser.make_features("2+4x"), 10, 1, 1, -1)
    features, move_count, focus_index, player_index, environment_state, last_action = state.decode_player(
        board, 1
    )
    assert str(state.parser.parse_features(features)).strip() == "2 + 4x"
    assert move_count == 10
    assert focus_index == 1
    assert player_index == 1
    assert environment_state == 1
    assert last_action == -1


def test_math_state_decode_player():
    state = EnvironmentState(128)
    for player_id in (1, -1):
        board = state.encode_board("4x+2")
        _, move_count, focus_index, player_index, environment_state, last_action = state.decode_player(
            board, player_id
        )
        assert move_count == 0
        # Focus is randomized between 0 and 2 to start
        assert focus_index >= 0 and focus_index <= 2
        assert player_id == player_index
        assert environment_state == 0
        assert last_action == -1


def test_math_state_get_canonical_board():
    state = EnvironmentState(128)
    parser = ExpressionParser()
    board = state.encode_board("4x+2")
    board = state.encode_player(board, -1, parser.make_features("2+4x"), 1, 1, 1, -1)
    board = state.get_canonical_board(board, -1)
    # The canonical board always represents the board from the same perspective, in this case
    # from the perspective of player 1. So player -1's canonical board will return the player -1
    # state when you decode player 1 from it.
    _, move_count, focus_index, player_index, environment_state, last_action = state.decode_player(board, 1)
    assert move_count == 1
    assert focus_index == 1
    assert player_index == -1
