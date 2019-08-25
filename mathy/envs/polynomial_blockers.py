from typing import Any, Dict, List, Optional, Type

from numpy.random import randint

from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnvProblem
from ..mathy_env_state import MathyEnvState
from ..rules import (
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
)
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
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
            blockers = randint(1, 3)
            text, complexity = move_around_blockers_one(blockers)
        elif params.difficulty == MathyEnvDifficulty.normal:
            blockers = randint(2, 5)
            text, complexity = move_around_blockers_two(blockers)
        elif params.difficulty == MathyEnvDifficulty.hard:
            blockers = randint(3, 8)
            text, complexity = move_around_blockers_two(blockers)
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity - 1, MODE_SIMPLIFY_POLYNOMIAL)
