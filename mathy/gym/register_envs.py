from gym.envs.registration import register

register(id="mathy-poly-easy-v0", entry_point="mathy.gym.polynomials:PolynomialsEasy")
register(
    id="mathy-poly-normal-v0", entry_point="mathy.gym.polynomials:PolynomialsNormal"
)
register(id="mathy-poly-hard-v0", entry_point="mathy.gym.polynomials:PolynomialsHard")

register(
    id="mathy-poly-blockers-easy-v0",
    entry_point="mathy.gym.polynomials:PolynomialBlockersEasy",
)
register(
    id="mathy-poly-blockers-normal-v0",
    entry_point="mathy.gym.polynomials:PolynomialBlockersNormal",
)
register(
    id="mathy-poly-blockers-hard-v0",
    entry_point="mathy.gym.polynomials:PolynomialBlockersHard",
)

register(id="mathy-binomial-easy-v0", entry_point="mathy.gym.binomials:BinomialsEasy")
register(
    id="mathy-binomial-normal-v0", entry_point="mathy.gym.binomials:BinomialsNormal"
)
register(id="mathy-binomial-hard-v0", entry_point="mathy.gym.binomials:BinomialsHard")
