from multiprocessing import cpu_count


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        pass

    def get_initial_state(self):
        """
        Returns:
            startBoard: a representation of the env_state (ideally this is the form
                        that will be the input to your neural network)
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

    def get_next_state(self, env_state, player, action, searching=False):
        """
        Input:
            env_state:     current env_state
            player:    current player (1 or -1)
            action:    action taken by current player
            searching: boolean set to True when called by MCTS

        Returns:
            nextBoard: env_state after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getValidMoves(self, env_state, player):
        """
        Input:
            env_state: current env_state
            player: current player

        Returns:
            validMoves: a binary vector of length self.get_agent_actions_count(), 1 for
                        moves that are valid from the current env_state and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, env_state, player, searching=False):
        """
        Input:
            env_state:     current env_state
            player:    current player (1 or -1)
            searching: boolean that is True when called by MCTS simulation

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """

    def getCanonicalForm(self, env_state, player):
        """
        Input:
            env_state: current env_state
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of env_state. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            env_state as is. When the player is black, we can invert
                            the colors and return the env_state.
        """
        pass

    def getSymmetries(self, env_state, pi):
        """
        Input:
            env_state: current env_state
            pi: policy vector of size self.get_agent_actions_count()

        Returns:
            symmForms: a list of [(env_state,pi)] where each tuple is a symmetrical
                       form of the env_state and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
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
