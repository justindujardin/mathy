from ..poly_haystack_like_terms import PolyHaystackLikeTerms
from ...types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Identify like terms in a haystack
#


class GymPolynomialLikeTermsHaystack(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(GymPolynomialLikeTermsHaystack, self).__init__(
            env_class=PolyHaystackLikeTerms,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialLikeTermsHaystackEasy(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs):
        super(PolynomialLikeTermsHaystackEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialLikeTermsHaystackNormal(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs):
        super(PolynomialLikeTermsHaystackNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialLikeTermsHaystackHard(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs):
        super(PolynomialLikeTermsHaystackHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-like-terms-haystack-easy-v0",
    entry_point="mathy.envs.gym:PolynomialLikeTermsHaystackEasy",
)
safe_register(
    id="mathy-poly-like-terms-haystack-normal-v0",
    entry_point="mathy.envs.gym:PolynomialLikeTermsHaystackNormal",
)
safe_register(
    id="mathy-poly-like-terms-haystack-hard-v0",
    entry_point="mathy.envs.gym:PolynomialLikeTermsHaystackHard",
)
