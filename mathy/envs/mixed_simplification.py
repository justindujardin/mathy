from typing import Any, Dict, List, Type

from ..mathy_env_state import MathyEnvState
from ..core.rules import (
    ConstantsSimplifyRule,
    VariableMultiplyRule,
    DistributiveFactorOutRule,
    BaseRule,
)
from ..agent.curriculum.problems import simplify_multiple_terms
from ..game_modes import MODE_SIMPLIFY_COMPLEX_TERM, MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnvProblem
from .polynomial_simplification import MathyPolynomialSimplificationEnv


class MathyMixedSimplificationEnv(MathyPolynomialSimplificationEnv):
    _counter = 0

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        if state.agent.problem_type == MODE_SIMPLIFY_COMPLEX_TERM:
            return [
                ConstantsSimplifyRule,
                VariableMultiplyRule,
                DistributiveFactorOutRule,
            ]
        # if not complex, must be poly simplification
        assert state.agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL

        return [ConstantsSimplifyRule, VariableMultiplyRule, DistributiveFactorOutRule]

    def problem_fn(self, params: Dict[str, Any] = None) -> MathyEnvProblem:

        # one
        # complex = 2
        # poly = 4

        # two
        # complex = 3
        # poly = 6

        # three
        complex = 4
        poly = 8

        config = params if params is not None else dict()
        num_terms = int(config.get("complex_difficulty", complex))
        simple = config.get("simple", False)
        self._counter += 1
        if self._counter % 2 != 0:
            # complex single-terms
            text, complexity = simplify_multiple_terms(
                num_terms,
                op="*",
                optional_var=simple,
                optional_var_probability=0.66,
                min_terms=2,
                inner_terms_scaling=0.1,
            )
            return MathyEnvProblem(text, complexity + 1, MODE_SIMPLIFY_COMPLEX_TERM)
        # polynomial simplification
        num_terms = int(config.get("poly_difficulty", poly))
        text, complexity = simplify_multiple_terms(num_terms)
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
