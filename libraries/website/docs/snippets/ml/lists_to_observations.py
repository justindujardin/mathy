from mathy import (
    MathyEnv,
    MathyEnvState,
    MathyObservation,
    envs,
    observations_to_window,
)

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]
observation: MathyObservation = env.state_to_observation(state, rnn_size=128)

# As many nodes as values
assert len(observation.nodes) == len(observation.values)
# Mask is number of nodes times number of actions
assert len(observation.mask) == len(observation.nodes) * env.action_size
# RNN states are the same size
assert len(observation.rnn_state_h) == 128
assert len(observation.rnn_state_c) == 128
assert len(observation.rnn_history_h) == 128
