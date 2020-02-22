import math
from typing import Any, List, Tuple

import numpy

from ..env import MathyEnv
from .policy_value_model import PolicyValueModel
from ..state import MathyEnvState, observations_to_window
from ..util import is_terminal_transition

EPS = 1e-8


class MCTS:
    """Monte-Carlo Tree Search is used for improving the actions selected
    for an agent by simulating a bunch of actions and returning the best
    based on the reward at the end of the simulated episodes."""

    env: MathyEnv
    # cpuct is a hyperparameter controlling the degree of exploration
    # (1.0 in Suragnair experiments.)
    cpuct: float
    num_mcts_sims: int
    # Set epsilon = 0 to disable dirichlet noise in root node.
    # e.g. for ExaminationRunner competitions
    epsilon: float
    dir_alpha: float

    def __init__(
        self,
        env: MathyEnv,
        model: PolicyValueModel,
        cpuct: float = 1.0,
        num_mcts_sims: int = 15,
        epsilon: float = 0.25,
        dir_alpha: float = 0.3,
    ):
        self.env = env
        self.model = model
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct
        self.dir_alpha = dir_alpha
        self.epsilon = epsilon

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times env_state s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores env.get_state_transition ended for env_state s
        self.Vs = {}  # stores env.get_valid_moves for env_state s

    def estimate_policy(
        self, env_state: MathyEnvState, temp: float = 1.0
    ) -> Tuple[List[float], float]:
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        env_state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
            value: the mean value of this state across all rollouts
        """
        probs: List[float]
        values: List[float] = []
        for _ in range(self.num_mcts_sims):
            values.append(self.search(env_state, True))
        value = numpy.asarray(values).mean()

        s = self.env.to_hash_key(env_state)
        counts = []
        for a in range(self.env.get_agent_actions_count(env_state)):
            if (s, a) in self.Nsa:
                counts.append(self.Nsa[(s, a)])
            else:
                counts.append(0)

        if temp == 0:
            bestA = numpy.argmax(counts)
            probs = [0.0] * len(counts)
            probs[bestA] = 1.0
            return probs, value

        counts = [x ** (1.0 / temp) for x in counts]
        count_sum = float(sum(counts))
        if count_sum == 0.0:
            # Arg, no valid moves picked from the tree! Let's go ahead and make each
            valids = numpy.array(self.env.get_valid_moves(env_state))
            return list(valids / valids.sum()), value
        probs = [x / float(count_sum) for x in counts]
        return probs, value

    def search(self, env_state: MathyEnvState, isRootNode=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current state
        """
        import tensorflow as tf

        s = self.env.to_hash_key(env_state)

        # if s not in self.Es:
        # print('calculating ending state for: {}'.format(s))
        self.Es[s] = self.env.get_state_transition(env_state, searching=True)
        if is_terminal_transition(self.Es[s]):
            # terminal node
            return self.Es[s].reward

        # This state does not have a predicted policy of value vector
        if s not in self.Ps:
            # leaf node
            valids = self.env.get_valid_moves(env_state)
            obs = env_state.to_observation(valids)
            observations = observations_to_window([obs]).to_inputs()
            out_policy, state_v = self.model.predict_next(observations)
            self.Ps[s] = out_policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            return state_v

        valids = self.Vs[s]
        cur_best = -float("inf")
        all_best: List[int] = []
        # add Dirichlet noise for root node. set epsilon=0 for ExaminationRunner
        # competitions of trained models
        add_noise = isRootNode and self.epsilon > 0
        if add_noise:
            moves = self.env.get_valid_moves(env_state)
            noise = numpy.random.dirichlet([self.dir_alpha] * len(moves))

        # pick the action with the highest upper confidence bound
        i = -1
        for a in range(len(valids)):
            if valids[a]:
                i += 1
                if (s, a) in self.Qsa:
                    q = self.Qsa[(s, a)]
                    n_s_a = self.Nsa[(s, a)]
                else:
                    q = 0
                    n_s_a = 0

                p = self.Ps[s][a]
                if add_noise:
                    p = (1 - self.epsilon) * p + self.epsilon * noise[i]

                u = q + self.cpuct * p * math.sqrt(self.Ns[s]) / (1 + n_s_a)

                if u > cur_best:
                    cur_best = u
                    del all_best[:]
                    all_best.append(a)
                elif u == cur_best:
                    all_best.append(a)

        a = numpy.random.choice(all_best)
        next_s, _, _ = self.env.get_next_state(env_state, a, searching=True)
        state_v = self.search(next_s)

        # state key for next state
        state_key = (s, a)
        if state_key in self.Qsa:
            self.Qsa[state_key] = (
                self.Nsa[state_key] * self.Qsa[state_key] + state_v
            ) / (self.Nsa[state_key] + 1)
            self.Nsa[state_key] += 1

        else:
            self.Qsa[state_key] = state_v
            self.Nsa[state_key] = 1

        self.Ns[s] += 1
        return state_v
