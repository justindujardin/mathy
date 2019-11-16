from gym.envs.registration import register

from ..polynomial_blockers import MathyPolynomialBlockersEnv
from ...types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv

#
# Commute + simplify with blockers
#


class GymPolynomialBlockers(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(GymPolynomialBlockers, self).__init__(
            env_class=MathyPolynomialBlockersEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialBlockersEasy(GymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialBlockersNormal(GymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialBlockersHard(GymPolynomialBlockers):
    def __init__(self, **kwargs):
        super(PolynomialBlockersHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


register(
    id="mathy-poly-blockers-easy-v0",
    entry_point="mathy.envs.gym:PolynomialBlockersEasy",
)
register(
    id="mathy-poly-blockers-normal-v0",
    entry_point="mathy.envs.gym:PolynomialBlockersNormal",
)
register(
    id="mathy-poly-blockers-hard-v0",
    entry_point="mathy.envs.gym:PolynomialBlockersHard",
)
