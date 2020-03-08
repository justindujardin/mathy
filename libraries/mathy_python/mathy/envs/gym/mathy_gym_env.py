import math
from typing import List, Optional, Type

import gym
import numpy as np
from gym.envs.registration import register

from ...core.expressions import MathExpression
from ...core.rule import ExpressionChangeRule
from ...env import MathyEnv
from ...state import (
    MathyEnvState,
    MathyObservation,
    observations_to_window,
)
from ...types import MathyEnvProblemArgs, MathyEnvProblem
from ...util import is_terminal_transition
from .masked_discrete import MaskedDiscrete


class MathyGymEnv(gym.Env):
    """A small wrapper around Mathy envs to allow them to work with OpenAI Gym. The
    agents currently use this env wrapper, but it could be dropped in the future."""

    metadata = {"render.modes": ["terminal", "attention"]}
    mathy: MathyEnv
    state: Optional[MathyEnvState]
    problem: Optional[MathyEnvProblem]
    env_class: Type[MathyEnv]
    env_problem_args: Optional[MathyEnvProblemArgs]
    last_action: int
    last_reward: float
    last_change: Optional[ExpressionChangeRule]

    def __init__(
        self,
        env_class: Type[MathyEnv] = MathyEnv,
        env_problem_args: Optional[MathyEnvProblemArgs] = None,
        **env_kwargs,
    ):
        self.mathy = env_class(**env_kwargs)
        self.env_class = env_class
        self.env_problem_args = env_problem_args
        if self.env_problem_args is not None and not isinstance(
            self.env_problem_args, MathyEnvProblemArgs
        ):
            raise ValueError("Problem args must be a MathyEnvProblemArgs instance")

        self.last_action = -1
        self.last_change = None
        self.last_reward = 0.0
        max_nodes = 1024
        max_actions = self.mathy.action_size
        vector_width = 1  # a single number
        self.state = None
        self.problem = None
        self.vectors_shape = (max_nodes, vector_width)
        self.action_space = MaskedDiscrete(max_actions, [1] * max_actions)

    @property
    def action_size(self) -> int:
        if self.state is not None:
            return self.mathy.get_agent_actions_count(self.state)
        return self.mathy.action_size

    def step(self, action):
        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        observation = self._observe(self.state)
        done = is_terminal_transition(transition)
        info = {"transition": transition, "done": done}
        if done:
            info["win"] = transition.reward > 0.0
        self.last_action = action
        self.last_change = change
        self.last_reward = round(float(transition.reward), 4)
        return self.state.to_np(), transition.reward, done, info

    def reset_with_input(self, problem_text: str, max_moves=16):
        self.last_action = -1
        self.last_change = None
        self.last_reward = 0.0
        # If the episode is being reset because it ended, assert the validity
        # of the last problem outcome
        if self.state is not None:
            self.mathy.finalize_state(self.state)
        self.state, self.problem = self.mathy.get_initial_state(self.env_problem_args)
        self.state = MathyEnvState(problem=problem_text, max_moves=max_moves)
        self._observe(self.state)
        return self.state

    def reset(self):
        self.last_action = -1
        self.last_change = None
        self.last_reward = 0.0
        # If the episode is being reset because it ended, assert the validity
        # of the last problem outcome
        if self.state is not None:
            problem_text = self.state.agent.history[0].raw
            max_moves = self.state.max_moves
            self.mathy.finalize_state(self.state)
            self.state = MathyEnvState(problem=problem_text, max_moves=max_moves)
        else:
            self.state, self.problem = self.mathy.get_initial_state(
                self.env_problem_args
            )
        self._observe(self.state)
        return self.state

    def initial_window(self):
        """return an n-step set of observations for initializing the env"""
        state, _ = self.mathy.get_initial_state(self.env_problem_args)
        return observations_to_window([self.mathy.state_to_observation(state)])

    def _observe(self, state: MathyEnvState) -> MathyObservation:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state. """
        action_mask = self.mathy.get_valid_moves(state)
        observation = self.mathy.state_to_observation(state)
        self.action_space.n = self.mathy.get_agent_actions_count(state)
        self.action_space.mask = action_mask
        return observation

    def render(self, mode="terminal", data=None):
        action_name = "initial"
        token_index = -1
        if self.last_action != -1 and self.last_change is not None:
            action_index, token_index = self.mathy.get_action_indices(self.last_action)
            action_name = self.mathy.rules[action_index].name
        else:
            print(f"Problem: {self.state.agent.problem}")
        self.mathy.print_state(
            self.state,
            action_name[:25].lower(),
            token_index,
            change=self.last_change,
            change_reward=self.last_reward,
        )


def safe_register(id: str, **kwargs):
    """Ignore re-register errors."""
    try:
        return register(id, **kwargs)
    except gym.error.Error:
        pass
