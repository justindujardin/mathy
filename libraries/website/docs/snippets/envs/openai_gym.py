#!pip install gym
import gym
from mathy_envs.gym import MathyGymEnv
from mathy_envs.state import MathyObservation

all_envs = gym.envs.registration.registry.all()  # type:ignore
# Filter to just mathy registered envs
mathy_envs = [e for e in all_envs if e.id.startswith("mathy-")]

assert len(mathy_envs) > 0

# Each env can be created and produce an initial observation without
# special configuration.
for gym_env_spec in mathy_envs:
    wrapper_env: MathyGymEnv = gym.make(gym_env_spec.id)  # type:ignore
    assert wrapper_env is not None
    observation: MathyObservation = wrapper_env.reset()
    assert isinstance(observation, MathyObservation)
    assert observation is not None
