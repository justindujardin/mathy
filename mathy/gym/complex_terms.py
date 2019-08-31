from ..envs.complex_term_simplification import MathyComplexTermSimplificationEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymComplexTerms(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty):
        super(MathyGymComplexTerms, self).__init__(
            env_class=MathyComplexTermSimplificationEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
        )


class ComplexTermsEasy(MathyGymComplexTerms):
    def __init__(self):
        super(ComplexTermsEasy, self).__init__(difficulty=MathyEnvDifficulty.easy)


class ComplexTermsNormal(MathyGymComplexTerms):
    def __init__(self):
        super(ComplexTermsNormal, self).__init__(difficulty=MathyEnvDifficulty.normal)


class ComplexTermsHard(MathyGymComplexTerms):
    def __init__(self):
        super(ComplexTermsHard, self).__init__(difficulty=MathyEnvDifficulty.hard)
