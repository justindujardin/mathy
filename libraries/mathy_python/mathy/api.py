from dataclasses import dataclass
from typing import Optional, Union

from .agents.base_config import BaseConfig
from .agents.episode_memory import EpisodeMemory
from .agents.fragile import SwarmConfig, swarm_solve
from .agents.policy_value_model import PolicyValueModel, load_policy_value_model
from .state import MathyEnvState


@dataclass
class MathyAPIModelState:
    config: BaseConfig
    model: PolicyValueModel


@dataclass
class MathyAPISwarmState:
    config: SwarmConfig


class Mathy:
    """The standard interface for working with Mathy models and agents."""

    state: Union[MathyAPIModelState, MathyAPISwarmState]

    def __init__(
        self,
        *,
        model_path: str = None,
        model: PolicyValueModel = None,
        config: Union[BaseConfig, SwarmConfig] = None,
        silent: bool = False,
    ):
        if model_path is not None:
            model, config = load_policy_value_model(model_path, silent=silent)
            self.state = MathyAPIModelState(model=model, config=config)
        elif model is not None and config is not None:
            if not isinstance(model, PolicyValueModel):
                raise ValueError("model must derive PolicyValueModel for compatibility")
            if not isinstance(config, BaseConfig):
                raise ValueError("config must be a BaseConfig instance")
            self.state = MathyAPIModelState(model=model, config=config)
        else:
            if config is None:
                config = SwarmConfig()
            if not isinstance(config, SwarmConfig):
                raise ValueError("config must be a SwarmConfig instance")
            self.state = MathyAPISwarmState(config=config)

    def simplify(
        self, *, model: str = "mathy_alpha_sm", problem: str, max_steps: int = 128,
    ) -> EpisodeMemory:
        if isinstance(self.state, MathyAPISwarmState):
            return self.simplify_swarm(problem=problem, max_steps=max_steps)
        assert isinstance(
            self.state, MathyAPIModelState
        ), f"unknown state type: {type(self.state)}!"
        return self.simplify_model(model=model, problem=problem, max_steps=max_steps)

    def simplify_swarm(self, *, problem: str, max_steps: int) -> EpisodeMemory:
        assert isinstance(self.state, MathyAPISwarmState), "not configured for swarm"
        return swarm_solve(problem, self.state.config)

    def simplify_model(
        self, *, model: str = "mathy_alpha_sm", problem: str, max_steps: int,
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
        assert isinstance(self.state, MathyAPIModelState), "not configured for model"
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
        selector = GreedyActionSelector(model=self.state.model, episode=0, worker_id=0)
        done = False
        while not done:
            env.render(last_action=last_action, last_reward=last_reward)
            window = episode_memory.to_window_observation(
                last_observation, window_size=self.state.config.prediction_window_size
            )
            action, value = selector.select(
                last_state=env.state,
                last_window=window,
                last_action=last_action,
                last_reward=last_reward,
            )
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
