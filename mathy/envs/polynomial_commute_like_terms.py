from typing import Any, List, Optional, Type

from numpy.random import randint
from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..helpers import TermEx, get_term_ex, get_terms
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..problems import combine_terms_in_place, commute_haystack, rand_bool
from ..rules import (
    AssociativeSwapRule,
    BaseRule,
    CommutativeSwapRule,
    DistributiveFactorOutRule,
)
from ..state import MathyEnvState, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .polynomial_simplification import MathyPolynomialSimplificationEnv


class MathyPolynomialCommuteLikeTermsEnv(MathyPolynomialSimplificationEnv):
    """A Mathy environment for moving like terms near each other to enable
    further simplification.

    This task is intended to test the model's ability to identify like terms
    in a large string of unlike terms and its ability to use the commutative
    swap rule to reorder the expression bringing the like terms close together.
    """

    def __init__(self, **kwargs):
        super(MathyPolynomialCommuteLikeTermsEnv, self).__init__(**kwargs)
        self.rule = DistributiveFactorOutRule()

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [CommutativeSwapRule, AssociativeSwapRule]

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
    ) -> Optional[time_step.TimeStep]:
        """If the expression has any nodes that the DistributiveFactorOut rule
        can be applied to, the problem is solved. """
        if self.rule.find_node(expression) is not None:
            return time_step.termination(features, self.get_win_signal(env_state))
        return None

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        """This task is to move two terms near each other, which requires
        as many actions as there are blocker nodes. The problem complexity
        is a direct measure of this value."""

        return problem.complexity * 2

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.commute.like.terms"

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        if params.difficulty == MathyEnvDifficulty.easy:
            blockers = randint(1, 3)
            powers = rand_bool(20)
            easy = rand_bool(50)
            text, _ = commute_haystack(
                commute_blockers=blockers,
                min_terms=4,
                max_terms=10,
                easy=easy,
                powers=powers,
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            blockers = randint(2, 4)
            powers = rand_bool(40)
            easy = rand_bool(25)
            text, _ = commute_haystack(
                min_terms=8, max_terms=12, easy=easy, powers=powers
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            blockers = randint(5, 10)
            powers = rand_bool(60)
            easy = rand_bool(5)
            text, _ = commute_haystack(
                min_terms=6, max_terms=12, easy=easy, powers=powers
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, blockers + 1, MODE_SIMPLIFY_POLYNOMIAL)
