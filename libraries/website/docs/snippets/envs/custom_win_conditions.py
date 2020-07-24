"""Custom environment with win conditions that are met whenever
two nodes are adjacent to each other that can have the distributive
property applied to factor out a common term """

from typing import Optional
from mathy_core import (
    MathExpression,
    rules,
)
from mathy import (
    MathyEnv,
    MathyEnvState,
    MathyObservation,
    is_terminal_transition,
    time_step,
)


class CustomWinConditions(MathyEnv):
    rule = rules.DistributiveFactorOutRule()

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
    ) -> Optional[time_step.TimeStep]:
        # If the rule can find any applicable nodes
        if self.rule.find_node(expression) is not None:
            # Return a terminal transition with reward
            return time_step.termination(features, self.get_win_signal(env_state))
        # None does nothing
        return None


env = CustomWinConditions()

# This state is not terminal because none of the nodes can have the distributive
# factoring rule applied to them.
state_one = MathyEnvState(problem="4x + y + 2x")
transition = env.get_state_transition(state_one)
assert is_terminal_transition(transition) is False

# This is a terminal state because the nodes representing "4x + 2x" can
# have the distributive factoring rule applied to them.
state_two = MathyEnvState(problem="4x + 2x + y")
transition = env.get_state_transition(state_two)
assert is_terminal_transition(transition) is True
