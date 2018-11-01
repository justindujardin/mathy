import math
import numpy as np

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(
        self, game, nnet, cpuct=1, num_mcts_sims=15, epsilon=0.25, dir_alpha=0.3
    ):
        self.game = game
        self.nnet = nnet
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct
        self.dir_alpha = dir_alpha
        # Set epsilon = 0 to disable dirichlet noise in root node.
        # e.g. for Arena competitions
        self.epsilon = epsilon

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times env_state s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for env_state s
        self.Vs = {}  # stores game.getValidMoves for env_state s

    def getActionProb(self, env_state, temp=1):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        env_state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.num_mcts_sims):
            self.search(env_state, True)

        s = self.game.to_hash_key(env_state)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.get_agent_actions_count())
        ]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        count_sum = float(sum(counts))
        if count_sum == 0.0:
            raise ValueError(
                "there have been no actions taken to derive probabilities from."
                "This usually means that the problem you generated was solved without"
                "taking any actions. Make sure to generate problems that take at least"
                "a few actions to complete\n"
                "state = {}".format(s)
            )
        probs = [x / float(count_sum) for x in counts]
        return probs

    def search(self, env_state, isRootNode=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current state
        """

        s = self.game.to_hash_key(env_state)

        if s not in self.Es:
            # print('calculating ending state for: {}'.format(s))
            self.Es[s] = self.game.getGameEnded(env_state, searching=True)
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        # This state does not have a predicted policy of value vector
        if s not in self.Ps:
            # leaf node
            input_data = env_state.to_numpy()
            self.Ps[s], v = self.nnet.predict(input_data)
            # print('calculating valid moves for: {}'.format(s))
            # print("v = {}".format(v))
            # print("Ps = {}".format(self.Ps[s].shape))
            valids = self.game.getValidMoves(env_state)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            # print("sum Ps = {}".format(sum_Ps_s))
            if sum_Ps_s > 0:
                # renormalize so values sum to 1
                self.Ps[s] /= sum_Ps_s
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float("inf")
        all_best = []
        e = self.epsilon
        # add Dirichlet noise for root node. set epsilon=0 for Arena competitions of trained models
        add_noise = isRootNode and e > 0
        if add_noise:
            moves = self.game.getValidMoves(env_state)
            noise = np.random.dirichlet([self.dir_alpha] * len(moves))

        # pick the action with the highest upper confidence bound
        i = -1
        for a in range(self.game.get_agent_actions_count()):
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
                    p = (1 - e) * p + e * noise[i]

                u = q + self.cpuct * p * math.sqrt(self.Ns[s]) / (1 + n_s_a)

                if u > cur_best:
                    cur_best = u
                    del all_best[:]
                    all_best.append(a)
                elif u == cur_best:
                    all_best.append(a)

        a = np.random.choice(all_best)

        next_s = self.game.get_next_state(env_state, a, searching=True)

        v = self.search(next_s)

        # state key for next state
        state_key = (s, a)
        if state_key in self.Qsa:
            self.Qsa[state_key] = (self.Nsa[state_key] * self.Qsa[state_key] + v) / (
                self.Nsa[state_key] + 1
            )
            self.Nsa[state_key] += 1

        else:
            self.Qsa[state_key] = v
            self.Nsa[state_key] = 1

        self.Ns[s] += 1
        return v
