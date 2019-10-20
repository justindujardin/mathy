from typing import Any, Dict, List, Optional, Type

from numpy.random import randint, uniform
from tf_agents.trajectories import time_step

from ..core.expressions import MathExpression
from ..features import calculate_term_grouping_distances
from ..game_modes import MODE_SIMPLIFY_POLYNOMIAL
from ..helpers import get_terms, has_like_terms, is_preferred_term_form
from ..mathy_env import MathyEnv, MathyEnvProblem
from ..problems import simplify_multiple_terms
from ..rules import (
    BaseRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
)
from ..state import MathyEnvState, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from ..util import GameRewards


class MathyPolynomialSimplificationEnv(MathyEnv):
    """A Mathy environment for simplifying polynomial expressions.

    NOTE: This environment only generates polynomial problems with
     addition operations. Subtraction, Multiplication and Division
     operators are excluded. This is a good area for improvement.
    """

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        return problem.complexity * 3

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.simplify"

    def get_lose_signal(self, env_state: MathyEnvState) -> float:
        """Subtract the normalized GC error from the lose signal
        so that the penalty is reduced when the agent is closer to
        winning."""

        _, error = calculate_term_grouping_distances(env_state.agent.problem)

        return GameRewards.LOSE - error

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
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

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        """Given a set of parameters to control term generation, produce
        a polynomial problem with (n) total terms divided among (m) groups
        of like terms. A few examples of the form: `f(n, m) = p`
        - (3, 1) = "4x + 2x + 6x"
        - (6, 4) = "4x + v^3 + y + 5z + 12v^3 + x"
        - (4, 2) = "3x^3 + 2z + 12x^3 + 7z"
        """
        if params.difficulty == MathyEnvDifficulty.easy:
            num_terms = randint(2, 5)
            scaling = uniform(0.35, 0.5)
            text, complexity = simplify_multiple_terms(
                num_terms,
                inner_terms_scaling=scaling,
                powers_probability=0.1,
                noise_probability=0.6,
                shuffle_probability=0.1,
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            num_terms = randint(5, 8)
            scaling = uniform(0.35, 0.65)
            text, complexity = simplify_multiple_terms(
                num_terms, shuffle_probability=0.45, inner_terms_scaling=scaling
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            num_terms = randint(9, 12)
            scaling = uniform(0.25, 0.75)
            text, complexity = simplify_multiple_terms(
                num_terms,
                shuffle_probability=0.5,
                powers_probability=0.8,
                inner_terms_scaling=scaling,
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
