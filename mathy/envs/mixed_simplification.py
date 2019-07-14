from typing import Any, Dict, List, Type

from ..mathy_env_state import MathyEnvState
from ..core.rules import (
    ConstantsSimplifyRule,
    VariableMultiplyRule,
    DistributiveFactorOutRule,
    BaseRule,
)
from .problems import simplify_multiple_terms
from ..game_modes import MODE_SIMPLIFY_COMPLEX_TERM, MODE_SIMPLIFY_POLYNOMIAL
from ..mathy_env import MathyEnvProblem
from .polynomial_simplification import MathyPolynomialSimplificationEnv


class MathyMixedSimplificationEnv(MathyPolynomialSimplificationEnv):
    _counter = 0

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        if state.agent.problem_type == MODE_SIMPLIFY_COMPLEX_TERM:
            return [ConstantsSimplifyRule, VariableMultiplyRule]
        # if not complex, must be poly simplification
        assert state.agent.problem_type == MODE_SIMPLIFY_POLYNOMIAL

        return [ConstantsSimplifyRule, DistributiveFactorOutRule]

    def problem_fn(self, params: Dict[str, Any] = None) -> MathyEnvProblem:

        config = params if params is not None else dict()
        challenge = int(config.get("difficulty", 1))
        if challenge == 1:
            complex = 2
            poly = 4
        elif challenge == 2:
            complex = 3
            poly = 5
        elif challenge == 3:
            complex = 4
            poly = 6
        elif challenge == 4:
            complex = 5
            poly = 7
        else:
            raise EnvironmentError(f"unknown difficulty: {challenge}")
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
            )
            return MathyEnvProblem(text, complexity + 2, MODE_SIMPLIFY_COMPLEX_TERM)
        # polynomial simplification
        num_terms = int(config.get("poly_difficulty", poly))
        text, complexity = simplify_multiple_terms(num_terms)
        return MathyEnvProblem(text, complexity, MODE_SIMPLIFY_POLYNOMIAL)
