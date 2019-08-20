from ..envs.polynomial_simplification import MathyPolynomialSimplificationEnv
from .mathy_gym_env import MathyGymEnv


class MathyGymPolynomials(MathyGymEnv):
    def __init__(self, difficulty: int = 3):
        super(MathyGymPolynomials, self).__init__(
            env_class=MathyPolynomialSimplificationEnv,
            env_problem_args={"difficulty": difficulty},
        )


class Polynomials03(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials03, self).__init__(difficulty=3)


class Polynomials04(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials04, self).__init__(difficulty=4)


class Polynomials05(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials05, self).__init__(difficulty=5)


class Polynomials06(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials06, self).__init__(difficulty=6)


class Polynomials07(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials07, self).__init__(difficulty=7)


class Polynomials08(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials08, self).__init__(difficulty=8)


class Polynomials09(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials09, self).__init__(difficulty=9)


class Polynomials10(MathyGymPolynomials):
    def __init__(self):
        super(Polynomials10, self).__init__(difficulty=10)
