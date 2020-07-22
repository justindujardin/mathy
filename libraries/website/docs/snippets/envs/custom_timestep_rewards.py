"""Environment with user-defined rewards per-timestep based on the
rule that was applied by the agent."""

from typing import List, Type

from mathy import MathyEnv, MathyEnvState
from mathy_core import BaseRule, rules


class CustomTimestepRewards(MathyEnv):
    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [rules.AssociativeSwapRule]

    def get_penalizing_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [rules.CommutativeSwapRule]


env = CustomTimestepRewards()
problem = "4x + y + 2x"
expression = env.parser.parse(problem)
state = MathyEnvState(problem=problem)

_, transition, _ = env.get_next_state(
    state, env.random_action(expression, rules.AssociativeSwapRule),
)
# Expect positive reward
assert transition.reward > 0.0

_, transition, _ = env.get_next_state(
    state, env.random_action(expression, rules.CommutativeSwapRule),
)
# Expect neagative reward
assert transition.reward < 0.0
