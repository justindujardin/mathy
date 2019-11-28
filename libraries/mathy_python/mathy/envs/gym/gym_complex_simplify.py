from gym.envs.registration import register

from ..complex_simplify import ComplexSimplify
from ...types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class GymComplexTerms(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(GymComplexTerms, self).__init__(
            env_class=ComplexSimplify,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class ComplexTermsEasy(GymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class ComplexTermsNormal(GymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class ComplexTermsHard(GymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


register(id="mathy-complex-easy-v0", entry_point="mathy.envs.gym:ComplexTermsEasy")
register(id="mathy-complex-normal-v0", entry_point="mathy.envs.gym:ComplexTermsNormal")
register(id="mathy-complex-hard-v0", entry_point="mathy.envs.gym:ComplexTermsHard")

