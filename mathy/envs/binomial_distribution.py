from typing import Any, List, Optional, Type

from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..mathy_env_state import MathyEnvState
from ..rules import (
    BaseRule,
    ConstantsSimplifyRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)
from ..rules.helpers import get_terms, has_like_terms, is_preferred_term_form
from ..types import MathyEnvProblemArgs, MathyEnvDifficulty
from ..problems import binomial_times_binomial, binomial_times_monomial


class MathyBinomialDistributionEnv(MathyEnv):
    """A Mathy environment for distributing pairs of binomials.

    The FOIL method is sometimes used to solve these types of problems, where
    FOIL is just the distributive property applied to two binomials connected
    with a multiplication."""

    def get_env_namespace(self) -> str:
        return "mathy.binomials.mulptiply"

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [ConstantsSimplifyRule, DistributiveMultiplyRule, VariableMultiplyRule]

    def transition_fn(
        self, env_state: MathyEnvState, expression: MathExpression, features: Any
    ) -> Optional[time_step.TimeStep]:
        """If there are no like terms."""
        if not has_like_terms(expression):
            term_nodes = get_terms(expression)
            is_win = True
            for term in term_nodes:
                if not is_preferred_term_form(term):
                    is_win = False
            if is_win:
                return time_step.termination(features, self.get_win_signal(env_state))
        return None

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        """Given a set of parameters to control term generation, produce
        2 binomials expressions connected by a multiplication. """
        if params.difficulty == MathyEnvDifficulty.easy:
            text, complexity = binomial_times_monomial(min_vars=2, max_vars=3)
        elif params.difficulty == MathyEnvDifficulty.normal:
            text, complexity = binomial_times_binomial(
                min_vars=2,
                max_vars=2,
                powers_proability=0.1,
                like_variables_probability=0.0,
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            text, complexity = binomial_times_binomial(
                min_vars=2,
                max_vars=3,
                simple_variables=False,
                powers_proability=0.8,
                like_variables_probability=0.8,
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
