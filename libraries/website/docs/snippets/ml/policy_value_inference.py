from mathy.envs.gym import MathyGymEnv
from colr import color

from mathy.cli import setup_tf_env
from mathy.agents.a3c import A3CConfig
from mathy.agents.action_selectors import A3CGreedyActionSelector
from mathy.state import observations_to_window, MathyObservation, MathyEnvState
from mathy.agents.policy_value_model import get_or_create_policy_model, PolicyValueModel
from mathy.agents.episode_memory import EpisodeMemory
from mathy.util import calculate_grouping_control_signal
from mathy.envs import PolySimplify

setup_tf_env()
env = PolySimplify()
args = A3CConfig(model_dir="/dev/null", verbose=True,)
episode_memory = EpisodeMemory()
env_state: MathyEnvState = env.get_initial_state()[0]
last_observation: MathyObservation = env.state_to_observation(
    env_state, rnn_size=args.lstm_units
)
last_text = env_state.agent.problem
last_action = -1
last_reward = -1

selector = A3CGreedyActionSelector(
    model=get_or_create_policy_model(args=args, env_actions=env.action_size,),
    episode=0,
    worker_id=0,
)

# Set RNN to 0 state for start of episode
selector.model.embedding.reset_rnn_state()

# Start with the "init" sequence [n] times
for i in range(args.num_thinking_steps_begin + 1):
    rnn_state_h = selector.model.embedding.state_h.numpy()
    rnn_state_c = selector.model.embedding.state_c.numpy()
    seq_start = env_state.to_start_observation(rnn_state_h, rnn_state_c)
    selector.model.call(observations_to_window([seq_start]).to_inputs())

done = False
while not done:
    # store rnn state for replay training
    rnn_state_h = selector.model.embedding.state_h.numpy()
    rnn_state_c = selector.model.embedding.state_c.numpy()
    last_rnn_state = [rnn_state_h, rnn_state_c]

    # named tuples are read-only, so add rnn state to a new copy
    last_observation = MathyObservation(
        nodes=last_observation.nodes,
        mask=last_observation.mask,
        values=last_observation.values,
        type=last_observation.type,
        time=last_observation.time,
        rnn_state=last_rnn_state,
        rnn_history=episode_memory.rnn_weighted_history(args.lstm_units),
    )
    window = episode_memory.to_window_observation(last_observation)
    action, value = selector.select(
        last_state=env_state,
        last_window=window,
        last_action=last_action,
        last_reward=last_reward,
        last_rnn_state=last_rnn_state,
    )
    observation, reward, done, _ = env.step(env_state, action)

    rnn_state_h = selector.model.embedding.state_h.numpy()
    rnn_state_c = selector.model.embedding.state_c.numpy()

    observation = MathyObservation(
        nodes=observation.nodes,
        mask=observation.mask,
        values=observation.values,
        type=observation.type,
        time=observation.time,
        rnn_state=[rnn_state_h, rnn_state_c],
        rnn_history=episode_memory.rnn_weighted_history(args.lstm_units),
    )

    new_text = env_state.agent.problem
    grouping_change = calculate_grouping_control_signal(
        last_text, new_text, clip_at_zero=args.clip_grouping_control
    )
    episode_memory.store(
        observation=last_observation,
        action=action,
        reward=reward,
        grouping_change=grouping_change,
        value=value,
    )
    if done:
        # Last timestep reward
        win = reward > 0.0
        env.render_state(args.print_mode, None)
        print(color(text="SOLVE" if win else "FAIL", fore="green" if win else "red",))
        break

    last_observation = observation
    last_action = action
    last_reward = reward
