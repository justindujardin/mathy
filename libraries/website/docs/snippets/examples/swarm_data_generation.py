#!pip install gym


from mathy.solver import SwarmConfig, mathy_swarm
from mathy_envs.state import MathyEnvState

# Which values do we want from the history tree?
history_names = ["states", "actions", "rewards"]

# Configure the swarm
swarm = mathy_swarm(SwarmConfig(history=True, history_names=history_names))

# Run the swarm to generate tree history
swarm.run()

# Sample random batches
random_batches = swarm.tree.iterate_nodes_at_random(batch_size=32, names=history_names)
total_set = set()
total_generated = 0
for states, actions, rewards in random_batches:
    texts = [MathyEnvState.from_np(s).agent.problem for s in states]
    total_generated += len(texts)
    unique = list(set(texts))
    total_set.update(unique)
best_state = MathyEnvState.from_np(swarm.walkers.states.best_state)
swarm.env._env._env.mathy.print_history(best_state)
print(f"Generated {total_generated} states, {len(total_set)} of which are unique")
print(f"Highest reward encountered: {swarm.walkers.states.best_reward}")
print(best_state.agent.problem)
