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
from ...environment_state import MathEnvironmentState
from ...math_game import MathGame
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
        self, game: MathGame, env_state: MathEnvironmentState, model: MathModel, history
    ):
        """Pick an action, take it, and return the next state.

        returns: A tuple of (new_env_state, train_example, terminal_results_or_none)
        """

        # Hold on to the episode example data for training the neural net
        state = env_state.clone()
        example_data = state.to_input_features()
        move_count = state.max_moves - state.agent.moves_remaining

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(move_count < self.explore_for_n_moves)
        pi = self.mcts.getActionProb(state, temp=temp)
        action = numpy.random.choice(len(pi), p=pi)
        # print("step - {} - {}".format(pi, action))

        # Calculate the next state based on the selected action
        next_state, transition = game.get_next_state(state, action)
        r = transition.reward
        is_done = transition.step_type == time_step.StepType.LAST

        example_text = next_state.agent.problem
        is_term = is_terminal_transition(transition)
        is_win = True if is_term and r > 0 else False
        # out_policy = pi
        out_policy = numpy.reshape(pi, (len(game.available_rules), -1))
        action_i, token_i = game.get_action_indices(
            game.parser.parse(state.agent.problem), action
        )
        history.append(
            [
                example_data,
                out_policy,
                r,
                state.agent.problem,
                next_state.agent.problem,
                action_i,
                token_i,
            ]
        )
        # Output a single training example for per-step training
        train_example = {
            "input": state.agent.problem,
            "output": next_state.agent.problem,
            "action": action_i,
            "token": token_i,
            "reward": float(r),
            "discounted": float(r),
            "policy": out_policy,
            "features": copy.deepcopy(example_data),
        }
        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, train_example, None
        normal_rewards = [x[2] for x in history]
        # print("initial rewards: {}".format(numpy.asarray(rewards)))
        rewards = list(discount(normal_rewards, game.discount))
        # print("discounted rewards: {}".format(numpy.asarray(rewards)))
        examples = []
        problem_id = uuid.uuid4().hex
        for i, x in enumerate(history):
            examples.append(
                {
                    "problem": problem_id,
                    "input": x[3],
                    "output": x[4],
                    "action": x[5],
                    "token": x[6],
                    "reward": float(normal_rewards[i]),
                    "discounted": float(rewards[i]),
                    "policy": x[1],
                    "features": x[0],
                }
            )
        episode_reward = sum(rewards)
        return next_state, train_example, (examples, episode_reward, is_win)
