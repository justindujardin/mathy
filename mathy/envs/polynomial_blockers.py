from typing import Any, Dict, List, Optional, Type

from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnvProblem
from ..mathy_env_state import MathyEnvState
from ..rules import (
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
)
from ..types import MathyEnvProblemArgs, MathyEnvDifficulty
from .polynomial_simplification import MathyPolynomialSimplificationEnv
from .problems import move_around_blockers_one, move_around_blockers_two


class MathyPolynomialBlockersEnv(MathyPolynomialSimplificationEnv):
    """A Mathy environment for polynomial problems that have a variable
    string of mismatched terms separating two like terms.

    The goal is to:
      1. Commute the like terms so they become siblings
      2. Combine the sibling like terms
    """

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [ConstantsSimplifyRule, DistributiveFactorOutRule, CommutativeSwapRule]

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        if params.difficulty == MathyEnvDifficulty.easy:
            text, complexity = move_around_blockers_one(1)
            # HACK: The complexity will be 3, but that's usually too
            #       many moves for this easy problem which is always
            #       something like "4x + 2y + 3x", requiring exactly
            #       one commute, factor, and simplify action.
            complexity = 1
        elif params.difficulty == MathyEnvDifficulty.normal:
            text, complexity = move_around_blockers_two(4)
        elif params.difficulty == MathyEnvDifficulty.hard:
            text, complexity = move_around_blockers_two(8)
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
