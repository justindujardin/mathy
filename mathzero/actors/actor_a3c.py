from ..environment_state import MathEnvironmentState
from ..math_game import MathGame
from ..model.math_model import MathModel


class ActorA3C:
    """A3C actor that updates its model continuously from experience throughout each episode"""

    def step(
        self, game: MathGame, env_state: MathEnvironmentState, model: MathModel, history
    ):
        """Pick an action, take it, and return the next state.

        returns: A tuple of (new_env_state, terminal_results_or_none)
        """
        raise NotImplementedError("TODO: implement A3C actor fn")
