from typing import Any, Dict, List, Optional, Type

from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..rules import (
    BaseRule,
    ConstantsSimplifyRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)
from ..rules.util import get_terms, has_like_terms, is_preferred_term_form
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from .problems import simplify_distributive_binomial
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..mathy_env_state import MathyEnvState


class MathyBinomialDistributionEnv(MathyEnv):
    """A Mathy environment for distributing pairs of binomials.

    The FOIL method is sometimes used to solve these types of problems, where
    FOIL is just the distributive property applied to two binomials connected
    with a multiplication."""

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

    def problem_fn(self, params: Dict[str, Any] = None) -> MathyEnvProblem:
        """Given a set of parameters to control term generation, produce
        2 binomials expressions connected by a multiplication."""
        config = params if params is not None else dict()
        if "difficulty" not in config:
            raise ValueError(
                "problem 'difficulty' must be provided as an integer value. "
                "The value is to represent the relative difficulty of the problem"
                " in this case it is the number of terms to generate"
            )

        difficulty = int(config["difficulty"])
        if difficulty < 4:
            text, complexity = simplify_distributive_binomial(min_vars=1, max_vars=2)
        elif difficulty == 4:
            text, complexity = simplify_distributive_binomial(min_vars=2, max_vars=2)
        elif difficulty == 5:
            text, complexity = simplify_distributive_binomial(
                min_vars=2, max_vars=2, powers_proability=0.8
            )
        elif difficulty == 6:
            text, complexity = simplify_distributive_binomial(min_vars=3, max_vars=3)
        else:
            text, complexity = simplify_distributive_binomial(
                min_vars=3, max_vars=4, simple_variables=False
            )

        return MathyEnvProblem(text, complexity + 3, MODE_SIMPLIFY_POLYNOMIAL)
