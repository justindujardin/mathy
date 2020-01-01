from ..poly_commute_like_terms import PolyCommuteLikeTerms
from ...types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Combine like terms without commuting
#


class GymPolynomialCommuteLikeTerms(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(GymPolynomialCommuteLikeTerms, self).__init__(
            env_class=PolyCommuteLikeTerms,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialCommuteLikeTermsEasy(GymPolynomialCommuteLikeTerms):
    def __init__(self, **kwargs):
        super(PolynomialCommuteLikeTermsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialCommuteLikeTermsNormal(GymPolynomialCommuteLikeTerms):
    def __init__(self, **kwargs):
        super(PolynomialCommuteLikeTermsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialCommuteLikeTermsHard(GymPolynomialCommuteLikeTerms):
    def __init__(self, **kwargs):
        super(PolynomialCommuteLikeTermsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-commute-easy-v0",
    entry_point="mathy.envs.gym:PolynomialCommuteLikeTermsEasy",
)
safe_register(
    id="mathy-poly-commute-normal-v0",
    entry_point="mathy.envs.gym:PolynomialCommuteLikeTermsNormal",
)
safe_register(
    id="mathy-poly-commute-hard-v0",
    entry_point="mathy.envs.gym:PolynomialCommuteLikeTermsHard",
)
