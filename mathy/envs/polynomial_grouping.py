from typing import Any, List, Optional, Type

from numpy.random import randint
from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..helpers import TermEx, get_term_ex, get_terms
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..problems import commute_haystack
from ..rules import AssociativeSwapRule, BaseRule, CommutativeSwapRule
from ..state import MathyEnvState, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs


class MathyPolynomialGroupingEnv(MathyEnv):
    """A Mathy environment for grouping polynomial terms that are like.

    The goal is to commute all the like terms so they become siblings as quickly as
    possible.
    """

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.group_like_terms"

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        return problem.complexity * 2

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
    ) -> Optional[time_step.TimeStep]:
        """If all like terms are siblings."""
        term_nodes = get_terms(expression)
        already_seen: set = set()
        current_term = ""
        # Iterate over each term in order and build a unique key to identify its
        # term likeness. For this we drop the coefficient from the term and use
        # only its variable/exponent to build keys.
        for term in term_nodes:
            ex: Optional[TermEx] = get_term_ex(term)
            if ex is None:
                raise ValueError("should this happen?")
            key = f"{ex.variable}{ex.exponent}"
            # If the key is in the "already seen and moved on" list then we've failed
            # to meet the completion criteria. e.g. the final x in "4x + 2y + x"
            if key in already_seen:
                return None
            if key != current_term:
                already_seen.add(current_term)
                current_term = key

        return time_step.termination(features, self.get_win_signal(env_state))

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        if params.difficulty == MathyEnvDifficulty.easy:
            blockers = randint(1, 3)
            text, _ = commute_haystack(
                commute_blockers=1, min_terms=3, max_terms=6, easy=True, powers=False
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            blockers = randint(2, 4)
            text, _ = commute_haystack(
                min_terms=5, max_terms=10, easy=False, powers=True
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            blockers = randint(3, 6)
            text, _ = commute_haystack(
                min_terms=11, max_terms=16, easy=False, powers=True
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, blockers + 2, MODE_SIMPLIFY_POLYNOMIAL)
