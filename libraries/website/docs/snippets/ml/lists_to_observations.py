from mathy import MathyEnv, MathyEnvState, MathyObservation, envs

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]
observation: MathyObservation = env.state_to_observation(state)

# As many nodes as values
assert len(observation.nodes) == len(observation.values)
# Mask is number of nodes times number of actions
assert len(observation.mask) == len(observation.nodes) * env.action_size
