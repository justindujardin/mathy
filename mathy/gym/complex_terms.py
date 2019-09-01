from ..envs.complex_term_simplification import MathyComplexTermSimplificationEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymComplexTerms(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(MathyGymComplexTerms, self).__init__(
            env_class=MathyComplexTermSimplificationEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class ComplexTermsEasy(MathyGymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class ComplexTermsNormal(MathyGymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class ComplexTermsHard(MathyGymComplexTerms):
    def __init__(self, **kwargs):
        super(ComplexTermsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )
