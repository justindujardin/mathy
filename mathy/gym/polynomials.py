from ..envs.polynomial_blockers import MathyPolynomialBlockersEnv
from ..envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymPolynomials(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty):
        super(MathyGymPolynomials, self).__init__(
            env_class=MathyPolynomialSimplificationEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
        )


class PolynomialsEasy(MathyGymPolynomials):
    def __init__(self):
        super(PolynomialsEasy, self).__init__(difficulty=MathyEnvDifficulty.easy)


class PolynomialsNormal(MathyGymPolynomials):
    def __init__(self):
        super(PolynomialsNormal, self).__init__(difficulty=MathyEnvDifficulty.normal)


class PolynomialsHard(MathyGymPolynomials):
    def __init__(self):
        super(PolynomialsHard, self).__init__(difficulty=MathyEnvDifficulty.hard)


#
# Commute blockers
#


class MathyGymPolynomialBlockers(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty):
        super(MathyGymPolynomialBlockers, self).__init__(
            env_class=MathyPolynomialBlockersEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
        )


class PolynomialBlockersEasy(MathyGymPolynomialBlockers):
    def __init__(self):
        super(PolynomialBlockersEasy, self).__init__(difficulty=MathyEnvDifficulty.easy)


class PolynomialBlockersNormal(MathyGymPolynomialBlockers):
    def __init__(self):
        super(PolynomialBlockersNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal
        )


class PolynomialBlockersHard(MathyGymPolynomialBlockers):
    def __init__(self):
        super(PolynomialBlockersHard, self).__init__(difficulty=MathyEnvDifficulty.hard)
