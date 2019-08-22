from gym.envs.registration import register

register(id="mathy-poly-03-v0", entry_point="mathy.gym.polynomials:Polynomials03")
register(id="mathy-poly-04-v0", entry_point="mathy.gym.polynomials:Polynomials04")
register(id="mathy-poly-05-v0", entry_point="mathy.gym.polynomials:Polynomials05")
register(id="mathy-poly-06-v0", entry_point="mathy.gym.polynomials:Polynomials06")
register(id="mathy-poly-07-v0", entry_point="mathy.gym.polynomials:Polynomials07")
register(id="mathy-poly-08-v0", entry_point="mathy.gym.polynomials:Polynomials08")
register(id="mathy-poly-09-v0", entry_point="mathy.gym.polynomials:Polynomials09")
register(id="mathy-poly-10-v0", entry_point="mathy.gym.polynomials:Polynomials10")

register(id="mathy-binomial-easy-v0", entry_point="mathy.gym.binomials:BinomialsEasy")
register(
    id="mathy-binomial-normal-v0", entry_point="mathy.gym.binomials:BinomialsNormal"
)
register(id="mathy-binomial-hard-v0", entry_point="mathy.gym.binomials:BinomialsHard")
