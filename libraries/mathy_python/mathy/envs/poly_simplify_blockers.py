from typing import Any, Dict, List, Optional, Type

from numpy.random import randint

from ..env import MathyEnvProblem
from ..state import MathyEnvState
from ..core.rule import BaseRule
from ..rules import (
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    AssociativeSwapRule,
)
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .poly_simplify import PolySimplify
from ..problems import gen_move_around_blockers_one, gen_move_around_blockers_two, rand_bool


class PolySimplifyBlockers(PolySimplify):
    """A Mathy environment for polynomial problems that have a variable
    string of mismatched terms separating two like terms.

    The goal is to:
      1. Commute the like terms so they become siblings
      2. Combine the sibling like terms
    """

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.commute_then_simplify"

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        hard_block = rand_bool()
        powers_probability = 0.5
        if params.difficulty == MathyEnvDifficulty.easy:
            powers_probability = 0.1
            blockers = randint(1, 3)
            hard_blockers = 1
        elif params.difficulty == MathyEnvDifficulty.normal:
            blockers = randint(2, 5)
            hard_blockers = randint(2, 3)
        elif params.difficulty == MathyEnvDifficulty.hard:
            blockers = randint(3, 7)
            hard_blockers = randint(2, 4)
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        if hard_block:
            text, complexity = gen_move_around_blockers_two(
                hard_blockers, powers_probability=powers_probability
            )
        else:
            text, complexity = gen_move_around_blockers_one(blockers, powers_probability)
        return MathyEnvProblem(text, complexity, self.get_env_namespace())
