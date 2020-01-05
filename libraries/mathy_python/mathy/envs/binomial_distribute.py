from typing import Any, List, Optional, Type

from numpy.random import randint, uniform

from .. import time_step
from ..core.expressions import MathExpression
from ..core.rule import BaseRule
from ..env import MathyEnv, MathyEnvProblem
from ..problems import (
    gen_binomial_times_binomial,
    gen_binomial_times_monomial,
    rand_bool,
)
from ..rules import (
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)
from ..state import MathyEnvState, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from ..util import get_terms, has_like_terms, is_preferred_term_form


class BinomialDistribute(MathyEnv):
    """A Mathy environment for distributing pairs of binomials.

    The FOIL method is sometimes used to solve these types of problems, where
    FOIL is just the distributive property applied to two binomials connected
    with a multiplication."""

    def get_env_namespace(self) -> str:
        return "mathy.binomials.mulptiply"

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
        2 binomials expressions connected by a multiplication. """
        if params.difficulty == MathyEnvDifficulty.easy:
            if rand_bool(50):
                text, complexity = gen_binomial_times_monomial(min_vars=2, max_vars=3)
            else:
                text, complexity = gen_binomial_times_binomial(
                    min_vars=2,
                    max_vars=3,
                    powers_probability=uniform(0.1, 0.4),
                    like_variables_probability=uniform(0.3, 0.7),
                )
        elif params.difficulty == MathyEnvDifficulty.normal:
            text, complexity = gen_binomial_times_binomial(
                min_vars=2,
                max_vars=2,
                powers_probability=uniform(0.2, 0.6),
                like_variables_probability=uniform(0.2, 0.5),
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            text, complexity = gen_binomial_times_binomial(
                min_vars=2,
                max_vars=3,
                simple_variables=False,
                powers_probability=uniform(0.4, 0.8),
                like_variables_probability=uniform(0.1, 0.3),
            )
            complexity += 2
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, complexity, self.get_env_namespace())
