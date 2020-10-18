from mathy_envs import MathyEnv, MathyEnvState, MathyObservation, envs

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]
observation: MathyObservation = env.state_to_observation(state)

# As many nodes as values
assert len(observation.nodes) == len(observation.values)
# Mask is a binary validity mask of size (num_rules, num_nodes)
assert len(observation.mask) == len(env.rules)
assert len(observation.mask[0]) == len(observation.nodes)
