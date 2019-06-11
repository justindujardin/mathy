import copy
import uuid
from typing import Any, List, NamedTuple, Optional, Tuple

import numpy

from ...mathy_env import MathyEnv
from ...types import MathyEnvObservation, MathyEnvEpisodeResult
from ...mathy_env_state import MathyEnvState
from ...util import discount, is_terminal_transition
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
        self,
        env: MathyEnv,
        env_state: MathyEnvState,
        model: MathModel,
        history: List[MathyEnvObservation],
    ) -> Tuple[MathyEnvState, MathyEnvObservation, Optional[MathyEnvEpisodeResult]]:
        """Pick an action, take it, and return the next state.

        returns: A tuple of (new_env_state, train_example, terminal_results_or_none)
        """

        # Hold on to the episode example data for training the neural net
        state = env_state.clone()
        pi_mask = env.get_valid_moves(state)
        example_data = state.to_input_features(pi_mask)
        move_count = state.max_moves - state.agent.moves_remaining

        # If the move_count is less than threshold, set temp = 1 else 0
        temp = int(move_count < self.explore_for_n_moves)
        pi = self.mcts.getActionProb(state, temp=temp)
        action = numpy.random.choice(len(pi), p=pi)
        # print("step - {} - {}".format(pi, action))

        # Calculate the next state based on the selected action
        next_state, transition = env.get_next_state(state, action)
        r = transition.reward
        is_term = is_terminal_transition(transition)
        is_win = True if is_term and r > 0 else False
        out_policy = pi
        out_policy = numpy.reshape(pi, (-1, len(env.actions))).tolist()
        pi_mask = numpy.reshape(pi_mask, (-1, len(env.actions))).tolist()
        action_i, token_i = env.get_action_indices(action)
        # Output a single training example for per-step training
        train_features = copy.deepcopy(example_data)
        train_features["policy_mask"] = pi_mask
        train_example = MathyEnvObservation(
            input=state.agent.problem,
            output=next_state.agent.problem,
            action=action_i,
            token=token_i,
            reward=float(r),
            discounted=float(r),
            policy=out_policy,
            features=train_features,
            problem="",
        )
        history.append(train_example)
        # Keep going if the reward signal is not terminal
        if not is_term:
            return next_state, train_example, None
        normal_rewards = [x.reward for x in history]
        rewards = list(discount(normal_rewards, env.discount))
        #
        episode_totals = numpy.cumsum(rewards)
        numpy.set_printoptions(precision=3, suppress=True)
        print(
            "normal, discounted, total \n{}".format(
                numpy.asarray(list(zip(normal_rewards, rewards, episode_totals)))
            )
        )
        final_history = []
        problem_id = uuid.uuid4().hex
        for i, x in enumerate(history):
            final_history.append(
                MathyEnvObservation(
                    input=x.input,
                    output=x.output,
                    action=x.action,
                    token=x.token,
                    reward=x.reward,
                    policy=x.policy,
                    features=x.features,
                    problem=problem_id,
                    discounted=float(rewards[i]),
                )
            )
        episode_reward = sum(rewards)
        return (
            next_state,
            train_example,
            MathyEnvEpisodeResult(final_history, episode_reward, is_win),
        )
