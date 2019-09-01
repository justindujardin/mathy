from ..envs.polynomial_blockers import MathyPolynomialBlockersEnv
from ..envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymPolynomials(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(MathyGymPolynomials, self).__init__(
            env_class=MathyPolynomialSimplificationEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialsEasy(MathyGymPolynomials):
    def __init__(self, **kwargs):
        super(PolynomialsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialsNormal(MathyGymPolynomials):
    def __init__(self, **kwargs):
        super(PolynomialsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialsHard(MathyGymPolynomials):
    def __init__(self, **kwargs):
        super(PolynomialsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


#
# Commute blockers
#


class MathyGymPolynomialBlockers(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(MathyGymPolynomialBlockers, self).__init__(
            env_class=MathyPolynomialBlockersEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialBlockersEasy(MathyGymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialBlockersNormal(MathyGymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialBlockersHard(MathyGymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )
