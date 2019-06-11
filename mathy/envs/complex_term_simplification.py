from typing import Any, Dict, List, Type

from ..core.rules import ConstantsSimplifyRule, VariableMultiplyRule, BaseRule
from ..mathy_env_state import MathyEnvState
from ..agent.curriculum.problems import simplify_multiple_terms
from ..game_modes import MODE_SIMPLIFY_COMPLEX_TERM
from ..mathy_env import MathyEnvProblem
from .polynomial_simplification import MathyPolynomialSimplificationEnv


class MathyComplexTermSimplificationEnv(MathyPolynomialSimplificationEnv):
    """A Mathy environment for simplifying complex terms (e.g. 4x^3 * 7y) inside of
    expressions. The goal is to simplify the complex term within the allowed number
    of environment steps.
    """

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [ConstantsSimplifyRule, VariableMultiplyRule]

    def problem_fn(self, params: Dict[str, Any] = None) -> MathyEnvProblem:
        """Given a set of parameters to control term generation, produce
        a complex term that has a simple representation that must be found.
        - "4x * 2y^2 * 7q"
        - "7j * z^6"
        - "x * 2y^7 * 8z * 2x"
        """
        config = params if params is not None else dict()
        num_terms = config.get("difficulty", 2)
        simple = config.get("simple", False)
        text, complexity = simplify_multiple_terms(
            num_terms, op="*", optional_var=simple, optional_var_probability=0.5
        )
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_COMPLEX_TERM)
