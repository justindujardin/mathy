from multiprocessing import cpu_count


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below.
    """

    def __init__(self):
        pass

    def get_initial_state(self):
        """
        Returns:
            env_state: a representation of the env_state
        """
        pass

    def get_agent_state_size(self):
        """
        Returns:
            (x,y): a tuple of env_state dimensions
        """
        pass

    def get_agent_state(self, env_state):
        return env_state[1]

    def get_agent_actions_count(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def get_next_state(self, env_state, action, searching=False):
        """
        Input:
            env_state:     current env_state
            action:    action taken
            searching: boolean set to True when called by MCTS

        Returns:
            state: env_state after applying action
        """
        pass

    def getValidMoves(self, env_state):
        """
        Input:
            env_state: current env_state

        Returns:
            validMoves: a binary vector of length self.get_agent_actions_count(), 1 for
                        moves that are valid from the current env_state, 0 for invalid moves
        """
        pass

    def getGameEnded(self, env_state, searching=False):
        """
        Input:
            env_state:     current env_state
            searching: boolean that is True when called by MCTS simulation

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost.
               
        """

    def getCanonicalForm(self, env_state):
        """
        Input:
            env_state: current env_state

        Returns:
            canonicalBoard: returns canonical form of env_state. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            env_state as is. When the player is black, we can invert
                            the colors and return the env_state.
        """
        # TODO: I think this is useless with single-actor system
        pass

    def to_hash_key(self, env_state):
        """
        Input:
            env_state: current env_state

        Returns:
            boardString: a quick conversion of env_state to a string format.
                         Required by MCTS for hashing.
        """
        pass

    def getGPUFraction(self):
        """
        Returns:
            gpu_fraction: the fraction of GPU memory to dedicate to the 
                          neural network for this game instance.
        """
        # NOTE: we double the CPU count to start out allocating smaller amounts of memory.
        #       This is because if we oversubscribe CUDA can throw failed to allocate errors
        #       with a bunch of workers. This way Tensorflow will grow the allocation per worker
        #       only as needed.
        return 1 / (cpu_count() * 1.5)
