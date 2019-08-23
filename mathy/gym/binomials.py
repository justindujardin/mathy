from ..envs.binomial_distribution import MathyBinomialDistributionEnv
from .mathy_gym_env import MathyGymEnv


class MathyGymBinomialDistribution(MathyGymEnv):
    def __init__(self, difficulty: int):
        super(MathyGymBinomialDistribution, self).__init__(
            env_class=MathyBinomialDistributionEnv,
            env_problem_args={"difficulty": difficulty},
        )


class BinomialsEasy(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsEasy, self).__init__(difficulty=4)


class BinomialsNormal(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsNormal, self).__init__(difficulty=5)


class BinomialsHard(MathyGymBinomialDistribution):
    def __init__(self):
        super(BinomialsHard, self).__init__(difficulty=6)

