#!pip install gym
import gymnasium as gym
from mathy_envs.gym import MathyGymEnv

all_envs = gym.registry.values()
# Filter to just mathy registered envs
mathy_envs = [e for e in all_envs if e.id.startswith("mathy-")]

assert len(mathy_envs) > 0

# Each env can be created and produce an initial observation without
# special configuration.
for gym_env_spec in mathy_envs:
    wrapper_env: MathyGymEnv = gym.make(gym_env_spec.id)  # type:ignore
    assert wrapper_env is not None
    observation = wrapper_env.reset()
    assert observation is not None
