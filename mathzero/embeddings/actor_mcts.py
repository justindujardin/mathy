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
from .math_model import MathModel
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
        rewards = list(discount(rewards))

        # Note that we insert negative(max_discounted_reward) so that normalizing
        # the reward will never make a WIN value that is less than 0 or a lose that is
        # greater than 0.
        # if is_win:
        #     anchor_discount = -numpy.max(discounts)
        # else:
        #     anchor_discount = abs(numpy.min(discounts))
        # # Compute the normalized values, and slice off the last (anchor) element
        # rewards = normalize_rewards(discounts + [anchor_discount])[:-1]

        # If we're losing, reverse the reward values so they get more negative as
        # the agent approaches the losing move.
        # if not is_win:
        #     numpy.flip(rewards)

        # Floating point is tricky, so sometimes the values near zero will flip
        # their signs, so force the appropriate sign. This is okay(?) because it's
        # only small value changes, e.g. (-0.001 to 0.001)
        if is_win:
            rewards = [min(abs(r) + 1e-3, 1.0) for r in rewards]
        else:
            rewards = [max(-abs(r) - 1e-3, -1) for r in rewards]
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
        episode_reward = rewards[0]
        return next_state, (examples, episode_reward, is_win)
