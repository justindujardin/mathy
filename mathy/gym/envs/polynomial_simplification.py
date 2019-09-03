from gym.envs.registration import register

from ...envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from ...types import MathyEnvDifficulty, MathyEnvProblemArgs
from ..mathy_gym_env import MathyGymEnv


class GymPolynomialSimplification(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(GymPolynomialSimplification, self).__init__(
            env_class=MathyPolynomialSimplificationEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialsEasy(GymPolynomialSimplification):
    def __init__(self, **kwargs):
        super(PolynomialsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialsNormal(GymPolynomialSimplification):
    def __init__(self, **kwargs):
        super(PolynomialsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialsHard(GymPolynomialSimplification):
    def __init__(self, **kwargs):
        super(PolynomialsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


register(id="mathy-poly-easy-v0", entry_point="mathy.gym.envs:PolynomialsEasy")
register(id="mathy-poly-normal-v0", entry_point="mathy.gym.envs:PolynomialsNormal")
register(id="mathy-poly-hard-v0", entry_point="mathy.gym.envs:PolynomialsHard")
