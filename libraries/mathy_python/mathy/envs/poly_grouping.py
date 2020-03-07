from typing import Any, List, Optional, Type

from numpy.random import randint

from .. import time_step
from ..core.expressions import MathExpression
from ..util import TermEx, get_term_ex, get_terms
from ..env import MathyEnv, MathyEnvProblem
from ..problems import gen_commute_haystack
from ..core.rule import BaseRule
from ..state import MathyEnvState, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs


class PolyGroupLikeTerms(MathyEnv):
    """A Mathy environment for grouping polynomial terms that are like.

    The goal is to commute all the like terms so they become siblings as quickly as
    possible.
    """

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.group_like_terms"

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        return problem.complexity * 4

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
                continue
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
            text, _ = gen_commute_haystack(
                commute_blockers=1, min_terms=5, max_terms=7, easy=True, powers=False
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            blockers = randint(2, 4)
            text, _ = gen_commute_haystack(
                min_terms=5, max_terms=10, easy=False, powers=True
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            blockers = randint(3, 6)
            text, _ = gen_commute_haystack(
                min_terms=11, max_terms=16, easy=False, powers=True
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, blockers + 2, self.get_env_namespace())
