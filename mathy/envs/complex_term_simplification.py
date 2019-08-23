from typing import Any, Dict, List, Optional, Type

from ..game_modes import MODE_SIMPLIFY_COMPLEX_TERM
from ..mathy_env import MathyEnvProblem
from ..mathy_env_state import MathyEnvState
from ..rules import BaseRule, ConstantsSimplifyRule, VariableMultiplyRule
from ..types import MathyEnvProblemArgs, MathyEnvDifficulty
from .polynomial_simplification import MathyPolynomialSimplificationEnv
from .problems import simplify_multiple_terms


class MathyComplexTermSimplificationEnv(MathyPolynomialSimplificationEnv):
    """A Mathy environment for simplifying complex terms (e.g. 4x^3 * 7y) inside of
    expressions. The goal is to simplify the complex term within the allowed number
    of environment steps.
    """

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [ConstantsSimplifyRule, VariableMultiplyRule]

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        """Given a set of parameters to control term generation, produce
        a complex term that has a simple representation that must be found.
        - "4x * 2y^2 * 7q"
        - "7j * z^6"
        - "x * 2y^7 * 8z * 2x"
        """

        if params.difficulty == MathyEnvDifficulty.easy:
            text, complexity = simplify_multiple_terms(2, op="*", optional_var=False)
        elif params.difficulty == MathyEnvDifficulty.normal:
            text, complexity = simplify_multiple_terms(
                4, op="*", optional_var=True, optional_var_probability=0.5
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            text, complexity = simplify_multiple_terms(8, op="*", optional_var=False)
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_COMPLEX_TERM)
