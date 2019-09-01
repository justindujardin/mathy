from ..envs.binomial_distribution import MathyBinomialDistributionEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymBinomialDistribution(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs):
        super(MathyGymBinomialDistribution, self).__init__(
            env_class=MathyBinomialDistributionEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class BinomialsEasy(MathyGymBinomialDistribution):
    def __init__(self, **kwargs):
        super(BinomialsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class BinomialsNormal(MathyGymBinomialDistribution):
    def __init__(self, **kwargs):
        super(BinomialsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class BinomialsHard(MathyGymBinomialDistribution):
    def __init__(self, **kwargs):
        super(BinomialsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )
