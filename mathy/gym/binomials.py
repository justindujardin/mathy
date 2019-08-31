from ..envs.binomial_distribution import MathyBinomialDistributionEnv
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv


class MathyGymBinomialDistribution(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty):
        super(MathyGymBinomialDistribution, self).__init__(
            env_class=MathyBinomialDistributionEnv,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
        )


class BinomialsEasy(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsEasy, self).__init__(difficulty=MathyEnvDifficulty.easy)


class BinomialsNormal(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsNormal, self).__init__(difficulty=MathyEnvDifficulty.normal)


class BinomialsHard(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsHard, self).__init__(difficulty=MathyEnvDifficulty.hard)
