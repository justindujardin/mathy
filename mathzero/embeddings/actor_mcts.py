import math
import time
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin
import numpy
from ..environment_state import MathEnvironmentState
from ..util import (
    REWARD_LOSE,
    REWARD_WIN,
    discount,
    is_terminal_transition,
    normalize_rewards,
)
from ..training.mcts import MCTS
from .math_game import MathGame
from ..core.expressions import MathExpression
from ..model.math_model import MathModel
from tf_agents.environments import time_step


class ActorMCTS:
    def __init__(self, mcts: MCTS, explore_for_n_moves=15):
        self.mcts = mcts
        self.explore_for_n_moves = explore_for_n_moves

    def step(
        self, game: MathGame, env_state: MathEnvironmentState, model: MathModel, history
    ):
        """Pick an action, take it, and return the next state.

        returns: A tuple of (new_env_state, terminal_results_or_none)
        """

        # Hold on to the episode example data for training the neural net
        state = env_state.clone()
        example_data = state.to_input_features()
        move_count = state.max_moves - state.agent.moves_remaining

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(move_count < self.explore_for_n_moves)
        pi = self.mcts.getActionProb(state, temp=temp)
        action = numpy.random.choice(len(pi), p=pi)

        # Calculate the next state based on the selected action
        next_state, transition = game.get_next_state(state, action)
        r = transition.reward
        is_done = transition.step_type == time_step.StepType.LAST

        example_text = next_state.agent.problem
        is_term = is_terminal_transition(transition)
        is_win = True if is_term and r > 0 else False
        history.append([example_data, pi, r, example_text])
        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, None
        rewards = [x[2] for x in history]
        # # flip all timestep rewards to positive (hurray, we won!)
        # if is_win:
        #     rewards = [abs(r) for r in rewards]
        rewards = list(discount(rewards, game.discount))
        examples = []
        for i, x in enumerate(history):
            examples.append(
                {
                    "reward": float(rewards[i]),
                    "before": x[3],
                    "policy": x[1],
                    "inputs": x[0],
                }
            )
        episode_reward = sum(rewards)
        return next_state, (examples, episode_reward, is_win)
