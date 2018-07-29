from ..math_state import MathState


def test_math_state():
    state = MathState(128)
    assert state is not None


def test_math_state_encode_player():
    state = MathState(128)
    board = state.encode_board("4x+2")
    board = state.encode_player(board, 1, "2+4x", 10, 1)
    text, move_count, focus_index, player_index = state.decode_player(board, 1)
    assert text.strip() == "2+4x"
    assert move_count == 10
    assert focus_index == 1
    assert player_index == 1


def test_math_state_decode_player():
    state = MathState(128)
    for player_id in (1, -1):
        board = state.encode_board("4x+2")
        text, move_count, focus_index, player_index = state.decode_player(
            board, player_id
        )
        assert text.strip() == "4x+2"
        assert move_count == 0
        assert focus_index == 0
        assert player_id == player_index


def test_math_state_get_canonical_board():
    state = MathState(128)
    board = state.encode_board("4x+2")
    board = state.encode_player(board, -1, "2+4x", 1, 1)
    board = state.get_canonical_board(board, -1)
    # The canonical board always represents the board from the same perspective, in this case
    # from the perspective of player 1. So player -1's canonical board will return the player -1
    # state when you decode player 1 from it.
    text, move_count, focus_index, player_index = state.decode_player(board, 1)
    assert text.strip() == "2+4x"
    assert move_count == 1
    assert focus_index == 1
    assert player_index == -1
