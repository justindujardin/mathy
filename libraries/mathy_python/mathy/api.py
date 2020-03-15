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
        self, *, model: str = "mathy_alpha_sm", problem: str, max_steps: int = 128,
    ) -> EpisodeMemory:
        """Simplify an input problem using the PolySimplify environment.
        
        # Arguments
        model (str): The input model to use for picking simplifying actions
        problem (str): The ascii math problem text, e.g. `-(4 + 2x) * 8 / 7y^(3 - 2)`
        max_steps (int): The maximum number of episode steps to allow the agent to take
            while solving the problem. Taking more than this is considered a failure.

        # Returns
        (EpisodeMemory): The stored episode memory containing the intermediate steps to get
            to the solution for the input problem.

        """
        import gym
        import tensorflow as tf
        from colr import color
        from .envs.gym import MathyGymEnv
        from .agents.action_selectors import GreedyActionSelector
        from .state import observations_to_window, MathyObservation
        from .agents.policy_value_model import PolicyValueModel

        environment = "poly"
        difficulty = "easy"
        episode_memory = EpisodeMemory()
        env: MathyGymEnv = gym.make(f"mathy-{environment}-{difficulty}-v0")
        last_observation: MathyObservation = env.reset_with_input(
            problem_text=problem, max_moves=max_steps
        )
        assert env.state is not None
        last_text = env.state.agent.problem
        last_action = -1
        last_reward = 0.0
        selector = GreedyActionSelector(model=self.model, episode=0, worker_id=0)
        done = False
        while not done:
            env.render(last_action=last_action, last_reward=last_reward)
            window = episode_memory.to_window_observation(
                last_observation, window_size=self.config.prediction_window_size
            )
            try:
                action, value = selector.select(
                    last_state=env.state,
                    last_window=window,
                    last_action=last_action,
                    last_reward=last_reward,
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
            new_text = env.state.agent.problem
            episode_memory.store(
                observation=last_observation, action=action, reward=reward, value=value,
            )
            if done:
                # Last timestep reward
                win = reward > 0.0
                env.render(last_action=last_action, last_reward=last_reward)
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
