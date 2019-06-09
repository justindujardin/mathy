import copy
import math
import time
import uuid
from multiprocessing import Array, Pool, Process, Queue, cpu_count
from random import shuffle
from sys import stdin

import numpy
from tf_agents.trajectories import time_step

from ...core.expressions import MathExpression
from ...mathy_env_state import MathyEnvState
from ...mathy_env import MathyEnv
from ...util import discount, is_terminal_transition, normalize_rewards
from ..controller import MathModel
from .mcts import MCTS


class ActorMCTS:
    """Take a step as the agent using MCTS.

    TODO: this is an abstraction that didn't end up seeming very useful. Perhaps
    remove it by pushing this code back into another part of the agent?"""

    def __init__(self, mcts: MCTS, explore_for_n_moves=15):
        self.mcts = mcts
        self.explore_for_n_moves = explore_for_n_moves

    def step(
        self, game: MathyEnv, env_state: MathyEnvState, model: MathModel, history
    ):
        """Pick an action, take it, and return the next state.

        returns: A tuple of (new_env_state, train_example, terminal_results_or_none)
        """

        # Hold on to the episode example data for training the neural net
        state = env_state.clone()
        pi_mask = game.get_valid_moves(state)
        example_data = state.to_input_features(pi_mask)
        move_count = state.max_moves - state.agent.moves_remaining

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(move_count < self.explore_for_n_moves)
        pi = self.mcts.getActionProb(state, temp=temp)
        action = numpy.random.choice(len(pi), p=pi)
        # print("step - {} - {}".format(pi, action))

        # Calculate the next state based on the selected action
        next_state, transition = game.get_next_state(state, action)
        r = transition.reward
        is_term = is_terminal_transition(transition)
        is_win = True if is_term and r > 0 else False
        out_policy = pi
        out_policy = numpy.reshape(pi, (-1, len(game.actions))).tolist()
        pi_mask = numpy.reshape(pi_mask, (-1, len(game.actions))).tolist()
        action_i, token_i = game.get_action_indices(action)
        # Output a single training example for per-step training
        train_features = copy.deepcopy(example_data)
        train_features["policy_mask"] = pi_mask
        train_example = {
            "input": state.agent.problem,
            "output": next_state.agent.problem,
            "action": action_i,
            "token": token_i,
            "reward": float(r),
            "discounted": float(r),
            "policy": out_policy,
            "features": train_features,
        }
        history.append(train_example)
        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, train_example, None
        normal_rewards = [x["reward"] for x in history]
        rewards = list(discount(normal_rewards, game.discount))
        #
        episode_totals = numpy.cumsum(rewards)
        numpy.set_printoptions(precision=3, suppress=True)
        print(
            "rewards: normal, discounted, total \n{}".format(
                numpy.asarray(list(zip(normal_rewards, rewards, episode_totals)))
            )
        )
        problem_id = uuid.uuid4().hex
        for i, x in enumerate(history):
            x["problem"] = problem_id
            x["discounted"] = float(rewards[i])
        episode_reward = sum(rewards)
        return next_state, train_example, (history, episode_reward, is_win)
