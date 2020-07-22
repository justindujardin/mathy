"""Environment with user-defined terminal rewards"""

from mathy import MathyEnvState, envs, is_terminal_transition
from mathy_core.rules import ConstantsSimplifyRule


class CustomEpisodeRewards(envs.PolySimplify):
    def get_win_signal(self, env_state: MathyEnvState) -> float:
        return 20.0

    def get_lose_signal(self, env_state: MathyEnvState) -> float:
        return -20.0


env = CustomEpisodeRewards()

# Win by simplifying constants and yielding a single simple term form
state = MathyEnvState(problem="(4 + 2) * x")
expression = env.parser.parse(state.agent.problem)
action = env.random_action(expression, ConstantsSimplifyRule)
out_state, transition, _ = env.get_next_state(state, action)
assert is_terminal_transition(transition) is True
assert transition.reward == 20.0
assert out_state.agent.problem == "6x"

# Lose by applying a rule with only 1 move remaining
state = MathyEnvState(problem="2x + (4 + 2) + 4x", max_moves=1)
expression = env.parser.parse(state.agent.problem)
action = env.random_action(expression, ConstantsSimplifyRule)
out_state, transition, _ = env.get_next_state(state, action)
assert is_terminal_transition(transition) is True
assert transition.reward == -20.0
assert out_state.agent.problem == "2x + 6 + 4x"
