#!pip install gym
import gym
import mathy.envs.gym
from mathy.state import MathyObservation

all_envs = gym.envs.registration.registry.all()
# Filter to just mathy registered envs
mathy_envs = [e for e in all_envs if e.id.startswith("mathy-")]

assert len(mathy_envs) > 0

# Each env can be created and produce an initial observation without
# special configuration.
for gym_env_spec in mathy_envs:
    wrapper_env: mathy.envs.gym.MathyGymEnv = gym.make(gym_env_spec.id)
    assert wrapper_env is not None
    observation: MathyObservation = wrapper_env.reset()
    assert isinstance(observation, MathyObservation)
    assert observation is not None
