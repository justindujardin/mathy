from typing import Any, Dict, Optional, List, Type

from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..core.rules import ConstantsSimplifyRule, DistributiveFactorOutRule, BaseRule
from ..core.util import get_terms, has_like_terms, is_preferred_term_form
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..mathy_env_state import MathyEnvState
from .problems import simplify_multiple_terms


class MathyPolynomialSimplificationEnv(MathyEnv):
    """A Mathy environment for simplifying polynomial expressions.

    NOTE: This environment only generates polynomial problems with
     addition operations. Subtraction, Multiplication and Division
     operators are excluded. This is a good area for improvement.
    """

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [ConstantsSimplifyRule, DistributiveFactorOutRule]

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
        a polynomial problem with (n) total terms divided among (m) groups
        of like terms. A few examples of the form: `f(n, m) = p`
        - (3, 1) = "4x + 2x + 6x"
        - (6, 4) = "4x + v^3 + y + 5z + 12v^3 + x"
        - (4, 2) = "3x^3 + 2z + 12x^3 + 7z"
        """
        config = params if params is not None else dict()
        if "difficulty" not in config:
            raise ValueError(
                "problem 'difficulty' must be provided as an integer value. "
                "The value is to represent the relative difficulty of the problem"
                " in this case it is the number of terms to generate"
            )
        num_terms = int(config["difficulty"])
        text, complexity = simplify_multiple_terms(num_terms)
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
