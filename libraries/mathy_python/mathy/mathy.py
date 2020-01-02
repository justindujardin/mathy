from typing import Optional

from .agents.base_config import BaseConfig
from .agents.episode_memory import EpisodeMemory
from .agents.policy_value_model import PolicyValueModel, load_policy_value_model


class Mathy:
    """The standard interface for working with Mathy models and agents."""

    config: BaseConfig
    model: PolicyValueModel

    def __init__(
        self,
        *,
        model_path: str = None,
        model: PolicyValueModel = None,
        config: BaseConfig = None,
        silent: bool = False,
    ):
        if model_path is not None:
            self.model, self.config = load_policy_value_model(model_path, silent=silent)
        elif model is not None and config is not None:
            if not isinstance(model, PolicyValueModel):
                raise ValueError("model must derive PolicyValueModel for compatibility")
            self.model = model
            self.config = config
        else:
            raise ValueError(
                "Either 'model_path' or ('model' and 'config') must be provided"
            )

    def simplify(
        self,
        *,
        model: str = "mathy_alpha_sm",
        problem: str,
        max_steps: int = 20,
        thinking_steps: int = 3,
    ) -> EpisodeMemory:
        """Simplify an input problem using the PolySimplify environment.
        
        # Arguments
        model (str): The input model to use for picking simplifying actions
        problem (str): The ascii math problem text, e.g. `-(4 + 2x) * 8 / 7y^(3 - 2)`
        max_steps (int): The maximum number of episode steps to allow the agent to take
            while solving the problem. Taking more than this is considered a failure.
        thinking_steps (int): The number of timesteps to look at the problem before attempting
            to solve it. These steps **do not** count toward the `max_steps` argument total.

        # Returns
        (EpisodeMemory): The stored episode memory containing the intermediate steps to get
            to the solution for the input problem.

        """
        import gym
        import tensorflow as tf
        from colr import color
        from .envs.gym import MathyGymEnv
        from .agents.action_selectors import A3CEpsilonGreedyActionSelector
        from .state import observations_to_window, MathyObservation
        from .agents.policy_value_model import PolicyValueModel
        from .util import calculate_grouping_control_signal

        environment = "poly"
        difficulty = "easy"
        episode_memory = EpisodeMemory()
        env: MathyGymEnv = gym.make(f"mathy-{environment}-{difficulty}-v0")
        last_observation: MathyObservation = env.reset_with_input(
            problem_text=problem, rnn_size=self.config.lstm_units, max_moves=max_steps
        )
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = -1

        selector = A3CEpsilonGreedyActionSelector(
            model=self.model, episode=0, worker_id=0, epsilon=0
        )

        # Set RNN to 0 state for start of episode
        selector.model.embedding.reset_rnn_state()

        # Start with the "init" sequence [n] times
        for i in range(self.config.num_thinking_steps_begin + 1):
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())
            seq_start = env.state.to_start_observation(rnn_state_h, rnn_state_c)
            selector.model.call(observations_to_window([seq_start]).to_inputs())

        done = False
        while not done:
            env.render(self.config.print_mode, None)
            # store rnn state for replay training
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())
            last_rnn_state = [rnn_state_h, rnn_state_c]

            # named tuples are read-only, so add rnn state to a new copy
            last_observation = MathyObservation(
                nodes=last_observation.nodes,
                mask=last_observation.mask,
                values=last_observation.values,
                type=last_observation.type,
                time=last_observation.time,
                rnn_state_h=rnn_state_h,
                rnn_state_c=rnn_state_c,
                rnn_history_h=episode_memory.rnn_weighted_history(
                    self.config.lstm_units
                )[0],
            )
            window = episode_memory.to_window_observation(last_observation)
            try:
                action, value = selector.select(
                    last_state=env.state,
                    last_window=window,
                    last_action=last_action,
                    last_reward=last_reward,
                    last_rnn_state=last_rnn_state,
                )
            except KeyboardInterrupt:
                print("Done!")
                return episode_memory
            except BaseException as e:
                print("Prediction failed with error:", e)
                print("Inputs to model are:", window)
                continue
            # Take an env step
            observation, reward, done, _ = env.step(action)
            rnn_state_h = tf.squeeze(selector.model.embedding.state_h.numpy())
            rnn_state_c = tf.squeeze(selector.model.embedding.state_c.numpy())
            observation = MathyObservation(
                nodes=observation.nodes,
                mask=observation.mask,
                values=observation.values,
                type=observation.type,
                time=observation.time,
                rnn_state_h=rnn_state_h,
                rnn_state_c=rnn_state_c,
                rnn_history_h=episode_memory.rnn_weighted_history(
                    self.config.lstm_units
                )[0],
            )

            new_text = env.state.agent.problem
            grouping_change = calculate_grouping_control_signal(
                last_text, new_text, clip_at_zero=self.config.clip_grouping_control
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
                env.render(self.config.print_mode, None)
                print(
                    color(
                        text="SOLVE" if win else "FAIL", fore="green" if win else "red",
                    )
                )
                break

            last_observation = observation
            last_action = action
            last_reward = reward
        return episode_memory
