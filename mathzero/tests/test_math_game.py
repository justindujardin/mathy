from ..math_game import MathGame
from ..environment_state import EnvironmentState


def test_math_game_init():
    game = MathGame()
    assert game is not None
    state = game.getInitBoard()
    assert state is not None
    # State is a numpy encoded ndarray
    assert state.shape is not None


def test_math_game_win_conditions():

    expectations = [
        ("4x^2", True),
        ("2", True),
        ("4x * 2", False),
        ("4x * 2x", False),
        ("4x + 2x", False),
    ]

    # Valid solutions but out of scope so they aren't counted as wins.
    # This works because the problem sets exclude this type of > 2 term
    # polynomial expressions
    out_of_scope_valid = [
        ("3x + 2y + 7", False),
    ]

    game = MathGame()
    state = EnvironmentState(MathGame.width)
    for text, is_win in expectations + out_of_scope_valid:
        board = state.encode_board(text)
        assert game.getGameEnded(board, 1) == int(is_win)
