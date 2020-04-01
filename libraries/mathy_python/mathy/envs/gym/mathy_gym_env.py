import math
from typing import List, Optional, Type, Union

import gym
import numpy as np
from gym.envs.registration import register

from ...core.expressions import MathExpression
from ...core.rule import ExpressionChangeRule
from ...env import MathyEnv
from ...state import MathyEnvState, MathyObservation, observations_to_window
from ...types import MathyEnvProblem, MathyEnvProblemArgs
from ...util import discount, is_terminal_transition, pad_array
from .masked_discrete import MaskedDiscrete


class MathyGymEnv(gym.Env):
    """A small wrapper around Mathy envs to allow them to work with OpenAI Gym. The
    agents currently use this env wrapper, but it could be dropped in the future. """

    mathy: MathyEnv
    state: Optional[MathyEnvState]
    _challenge: Optional[MathyEnvState]
    env_class: Type[MathyEnv]
    env_problem_args: Optional[MathyEnvProblemArgs]

    def __init__(
        self,
        env_class: Type[MathyEnv] = MathyEnv,
        env_problem_args: Optional[MathyEnvProblemArgs] = None,
        env_problem: Optional[str] = None,
        env_max_moves: int = 64,
        np_observation: bool = False,
        repeat_problem: bool = False,
        **env_kwargs,
    ):
        self.state = None
        self.repeat_problem = repeat_problem
        self.np_observation = np_observation
        self.mathy = env_class(**env_kwargs)
        self.env_class = env_class
        self.env_problem_args = env_problem_args
        if env_problem is not None:
            self._challenge = MathyEnvState(
                problem=env_problem, max_moves=env_max_moves
            )
        else:
            self._challenge, _ = self.mathy.get_initial_state(env_problem_args)
        self.action_space = MaskedDiscrete(self.action_size, [1] * self.action_size)

    @property
    def action_size(self) -> int:
        if self.state is not None:
            return self.mathy.get_agent_actions_count(self.state)
        return self.mathy.action_size

    def step(self, action: int):
        assert self.state is not None, "call reset() before stepping the environment"
        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        done = is_terminal_transition(transition)
        info = {
            "transition": transition,
            "done": done,
            "valid": change.result is not None,
        }
        if done:
            info["win"] = transition.reward > 0.0
        return self._observe(self.state), transition.reward, done, info

    def _observe(self, state: MathyEnvState) -> Union[MathyObservation, np.ndarray]:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state. """
        action_mask = self.mathy.get_valid_moves(state)
        observation = self.mathy.state_to_observation(state)
        self.action_space.n = self.mathy.get_agent_actions_count(state)
        self.action_space.mask = action_mask
        if self.np_observation:
            # convert mask to probabilities
            nodes = np.array(pad_array(observation.nodes, 512, 0))
            mask = np.array(pad_array(observation.mask, 512, 0))
            mask_sum = np.sum(mask)
            if mask_sum > 0.0:
                mask = mask / mask_sum
            return np.vstack((nodes, mask))
        return observation

    def reset(self):
        if self.state is not None:
            self.mathy.finalize_state(self.state)
        if self.repeat_problem:
            self.state = MathyEnvState.copy(self._challenge)
        else:
            self.state, self.problem = self.mathy.get_initial_state(
                self.env_problem_args
            )
        return self._observe(self.state)

    def reset_with_input(self, problem_text: str, max_moves=16):
        # If the episode is being reset because it ended, assert the validity
        # of the last problem outcome
        if self.state is not None:
            self.mathy.finalize_state(self.state)
        self.reset()
        self.state = MathyEnvState(problem=problem_text, max_moves=max_moves)
        return self._observe(self.state)

    def render(
        self,
        last_action: int = -1,
        last_reward: float = 0.0,
        last_change: Optional[ExpressionChangeRule] = None,
    ):
        assert self.state is not None, "call reset() before rendering the env"
        action_name = "initial"
        token_index = -1
        if last_action != -1:
            action_index, token_index = self.mathy.get_action_indices(last_action)
            action_name = self.mathy.rules[action_index].name
        else:
            print(f"Problem: {self.state.agent.problem}")
        self.mathy.print_state(
            self.state,
            action_name[:25].lower(),
            token_index,
            change=last_change,
            change_reward=last_reward,
        )


def safe_register(id: str, **kwargs):
    """Ignore re-register errors."""
    try:
        return register(id, **kwargs)
    except gym.error.Error:
        pass
