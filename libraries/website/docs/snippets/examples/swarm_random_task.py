#!pip install gymnasium

import random

import gymnasium as gym
from mathy.solver import SwarmConfig, swarm_solve
from mathy_envs.gym import MathyGymEnv

config = SwarmConfig(max_iters=10)
task = random.choice(["poly", "binomial", "complex"])
env: MathyGymEnv = gym.make(f"mathy-{task}-easy-v0")
_, problem = env.mathy.get_initial_state(env.env_problem_args)
swarm_solve(problem.text, config)
